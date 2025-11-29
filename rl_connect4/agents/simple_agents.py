import random
from typing import Optional

from envs.connect_four_env import ConnectFourEnv


class RandomAgent:
    """Samples uniformly from legal moves."""

    def select_action(self, env: ConnectFourEnv) -> int:
        legal = env._legal_moves()
        return random.choice(legal)


class HeuristicAgent:
    """
    Simple heuristic:
    1) Take a winning move if available.
    2) Block opponent immediate win if possible.
    3) Otherwise, pick the most central legal column.
    """

    def select_action(self, env: ConnectFourEnv) -> int:
        legal = env._legal_moves()
        if not legal:
            raise ValueError("No legal actions available.")

        # Current player piece id (1 or 2)
        cur_piece = env.current_player + 1
        opp_piece = 2 if cur_piece == 1 else 1

        winning_move = self._find_winning_move(env, legal, cur_piece)
        if winning_move is not None:
            return winning_move

        blocking_move = self._find_winning_move(env, legal, opp_piece)
        if blocking_move is not None:
            return blocking_move

        center = (env.cols - 1) / 2.0
        return min(legal, key=lambda c: abs(c - center))

    def _find_winning_move(
        self, env: ConnectFourEnv, legal, piece: int
    ) -> Optional[int]:
        snapshot = env.board.copy()
        try:
            for col in legal:
                row = self._first_empty_row(snapshot, col)
                if row is None:
                    continue
                env.board[row, col] = piece
                if env._check_winner(piece):
                    return col
                env.board[row, col] = snapshot[row, col]
        finally:
            env.board[:] = snapshot
        return None

    def _first_empty_row(self, board, col: int) -> Optional[int]:
        for r in range(board.shape[0] - 1, -1, -1):
            if board[r, col] == 0:
                return r
        return None
