import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ConnectFourEnv(gym.Env):
    """
    Custom Connect Four environment using a single-agent Gymnasium-style API,
    suitable for PPO in self-play.

    - Board: defaults to 6 rows x 7 columns, configurable.
    - Two players alternate turns; internal state tracks the current player.
    - Observation (dict):
        * "observation": (rows, cols, 2) binary planes
              plane 0: current player's pieces
              plane 1: opponent's pieces
        * "action_mask": (cols,) binary vector of legal columns for the current player
    - Action: Discrete(cols) -> which column to drop a piece into.
    - Reward (from the perspective of the player who just moved):
        +1  if that move creates a connect_n in a row
        -1  if the move is illegal (column full)
         0  otherwise (including draws).
      Episode terminates on win, draw, or illegal move.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 2}

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        connect_n: int = 4,
        render_mode: str | None = None,
        illegal_move_penalty: float = -30.0,
        win_reward: float = 40.0,
        draw_reward: float = 0.0,
        block_reward: float = 0.1,
        threat_reward: float = 0.05,
        seed: int | None = None,
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.connect_n = connect_n
        self.render_mode = render_mode
        self.illegal_move_penalty = illegal_move_penalty
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.block_reward = block_reward
        self.threat_reward = threat_reward
        self.loss_reward = -30.0  # terminal loss (from the mover's perspective)

        # 0 = empty, 1 = player 0, 2 = player 1
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 0  # 0 or 1

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.rows, self.cols, 2),
                    dtype=np.int8,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.cols,),
                    dtype=np.int8,
                ),
            }
        )
        self.action_space = spaces.Discrete(self.cols)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Precompute winning line coordinates for fast winner checks
        self._winning_lines = self._precompute_winning_lines()
        # Fixed directions used for reward shaping
        self._reward_directions = (
            (0, 1),   # right
            (1, 0),   # down
            (1, 1),   # down-right
            (-1, 1),  # up-right
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.board[:] = 0
        self.current_player = 0
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        legal_mask = self._legal_mask()
        legal_cols = np.flatnonzero(legal_mask)
        if action not in legal_cols:
            # Treat illegal move as a terminal loss
            reward = float(self.loss_reward)
            terminated = True
            truncated = False
            info = {"illegal_move": True}
            return self._get_obs(), reward, terminated, truncated, info

        row = self._get_drop_row(action)
        piece = self.current_player + 1  # 1 or 2

        self.board[row, action] = piece

        if self._check_winner(piece):
            reward = float(self.win_reward)
            terminated = True
            truncated = False
            info = {"winner": self.current_player, "loser_reward": float(self.loss_reward)}
        elif not self._legal_mask().any():  # draw (board full, no legal moves)
            reward = float(self.draw_reward)
            terminated = True
            truncated = False
            info = {"draw": True}
        else:
            reward = 0.0
            reward += self._compute_line_reward(piece)
            reward += self._compute_column_reward(action)
            terminated = False
            truncated = False
            info = {}

        self.current_player = 1 - self.current_player

        observation = self._get_obs()
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Simple text render (for debugging in terminal).
        Returns a string; use: print(env.render()).
        """
        board_str = self._board_to_string()
        if self.render_mode == "ansi" or self.render_mode is None:
            return board_str
        raise NotImplementedError(f"Unsupported render_mode: {self.render_mode}")

    def close(self):
        return

    # ---------- Internal helpers ----------

    def _get_obs(self):
        """
        Build observation from the *current* player's perspective.
        Plane 0 = current player's pieces, Plane 1 = opponent's pieces.
        """
        cur = self.current_player + 1
        opp = 2 if cur == 1 else 1

        board = self.board
        cur_plane = (board == cur).astype(np.int8)
        opp_plane = (board == opp).astype(np.int8)
        obs = np.stack([cur_plane, opp_plane], axis=-1)

        mask = self._legal_mask()

        return {"observation": obs, "action_mask": mask}

    def _legal_moves(self):
        """
        A move is legal if the top cell of the column is empty.
        """
        return list(np.flatnonzero(self._legal_mask()))

    def action_masks(self):
        """
        Compatibility helper for mask-aware algorithms (e.g., sb3-contrib MaskablePPO).
        Returns a boolean mask of legal actions.
        """
        return self._get_obs()["action_mask"].astype(bool)

    def _get_drop_row(self, col):
        """
        Find the lowest empty row in the given column.
        """
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        raise ValueError(f"Column {col} is full; no drop row available.")

    def _check_winner(self, piece):
        """
        Check horizontal, vertical and both diagonal directions for connect_n
        of the given 'piece' value (1 or 2).
        """
        board = self.board
        for line in self._winning_lines:
            if all(board[r, c] == piece for r, c in line):
                return True
        return False

    def _count_direction(self, r, c, dr, dc, piece):
        """
        Count contiguous pieces matching `piece` starting at (r, c)
        moving in direction (dr, dc).
        """
        count = 0
        row, col = r, c
        for _ in range(self.connect_n):
            if 0 <= row < self.rows and 0 <= col < self.cols and self.board[row, col] == piece:
                count += 1
                row += dr
                col += dc
            else:
                break
        return count

    def _board_to_string(self):
        chars = {0: ".", 1: "X", 2: "O"}
        rows = []
        for r in range(self.rows):
            rows.append(" ".join(chars[int(x)] for x in self.board[r]))
        return "\n".join(rows)

    def _count_immediate_wins(self, piece: int) -> int:
        """
        Count how many legal columns would produce an immediate win
        for the given piece if played now.
        """
        wins = 0
        legal_mask = self._legal_mask()
        for col in np.flatnonzero(legal_mask):
            row = self._get_drop_row(col)
            self.board[row, col] = piece
            if self._check_winner(piece):
                wins += 1
            self.board[row, col] = 0
        return wins

    def _compute_line_reward(self, piece: int) -> float:
        """
        Reward contiguous lines of the given piece:
            length 1 -> +1
            length 2 -> +2
            length 3 -> +6
        Counts horizontal, vertical, diag down-right, diag up-right.
        Avoids double-counting by only starting runs where the previous cell
        in that direction is out of bounds or a different piece.
        """
        reward = 0.0
        board = self.board
        rows = self.rows
        cols = self.cols
        for r in range(rows):
            for c in range(cols):
                if board[r, c] != piece:
                    continue
                for dr, dc in self._reward_directions:
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < rows and 0 <= prev_c < cols and board[prev_r, prev_c] == piece:
                        continue  # not the start of a run
                    length = 0
                    rr, cc = r, c
                    while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == piece:
                        length += 1
                        rr += dr
                        cc += dc
                    if length == 1:
                        reward += 1.0
                    elif length == 2:
                        reward += 2.0
                    elif length == 3:
                        reward += 6.0
        return reward

    def _compute_column_reward(self, col: int) -> float:
        """
        Column-centric bonus per Taylor & Stella (2024):
            columns 1 or 5 (0-based) -> +1
            columns 2 or 4           -> +2
            column 3                 -> +4
        """
        if col in (1, 5):
            return 1.0
        if col in (2, 4):
            return 2.0
        if col == 3:
            return 4.0
        return 0.0

    def _precompute_winning_lines(self):
        """
        Precompute all connect_n line coordinate sets for fast winner checks.
        """
        lines = []
        r, c, n = self.rows, self.cols, self.connect_n
        # Horizontal
        for row in range(r):
            for col in range(c - n + 1):
                lines.append([(row, col + k) for k in range(n)])
        # Vertical
        for row in range(r - n + 1):
            for col in range(c):
                lines.append([(row + k, col) for k in range(n)])
        # Diagonal down-right
        for row in range(r - n + 1):
            for col in range(c - n + 1):
                lines.append([(row + k, col + k) for k in range(n)])
        # Diagonal up-right
        for row in range(n - 1, r):
            for col in range(c - n + 1):
                lines.append([(row - k, col + k) for k in range(n)])
        return lines

    def _legal_mask(self) -> np.ndarray:
        """
        Fast legal action mask: top cell empty => legal.
        """
        return (self.board[0] == 0).astype(np.int8)
