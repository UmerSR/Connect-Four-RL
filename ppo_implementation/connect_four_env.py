import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.signal import convolve2d
from typing import Optional, Tuple, Dict, Any

class ConnectFourEnv(gym.Env):
    """
    A specific implementation of Connect Four for Self-Play Reinforcement Learning.
    
    Representation:
        - Board: 6x7 Grid
        - Players: 1 (Current Agent), -1 (Opponent)
        - Observation: (3, 6, 7) Tensor
            - Ch0: Current Player's pieces (1 if present, 0 else)
            - Ch1: Opponent's pieces (1 if present, 0 else)
            - Ch2: Legal Move Mask (1 if column open, 0 else) broadcasted to grid
            
    Dynamics:
        - Gravity is handled via a 'heights' array for O(1) drop calculation.
        - Win checking is handled via convolution for O(1) complexity relative to board state.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Game Constants
        self.ROWS = 6
        self.COLS = 7
        self.WIN_LENGTH = 4
        
        # Action Space: Drop in one of 7 columns
        self.action_space = spaces.Discrete(self.COLS)
        
        # Observation Space: 3 Channels x 6 Rows x 7 Cols
        # Channel 0: "My" pieces
        # Channel 1: "Opponent" pieces
        # Channel 2: Valid moves mask (or simple turn indicator)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.ROWS, self.COLS), dtype=np.float32
        )
        
        self.render_mode = render_mode
        
        # Internal State
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.heights = np.zeros(self.COLS, dtype=np.int8) # Tracks next empty row per col
        self.player_turn = 1 # 1 or -1
        self.steps = 0
        
        # Pre-compute Win Kernels (Optimization for Speed)
        self._init_kernels()

    def _init_kernels(self):
        """Creates convolution kernels for detecting 4-in-a-row efficiently."""
        self.kernels = []
        
        # Horizontal
        self.kernels.append(np.ones((1, 4), dtype=np.int8))
        # Vertical
        self.kernels.append(np.ones((4, 1), dtype=np.int8))
        # Diagonal (Top-left to Bottom-right)
        self.kernels.append(np.eye(4, dtype=np.int8))
        # Anti-Diagonal (Top-right to Bottom-left)
        self.kernels.append(np.fliplr(np.eye(4, dtype=np.int8)))

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
            super().reset(seed=seed)
            
            self.board.fill(0)
            self.heights.fill(0)
            self.steps = 0
            
            # Standard Connect Four: Player 1 always starts
            self.player_turn = 1 
            
            # FIX: Added 'legal_moves' to info so the PPO script doesn't crash on startup
            info = {
                "turn": self.player_turn,
                "legal_moves": self._get_legal_moves_mask()
            }
            
            return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one PLY (one move by one player).
        Note: In self-play, the 'loop' calling this env must handle the alternating
        perspective. This env simply accepts a move for 'self.player_turn'.
        """
        
        # 1. Validity Check
        if self.heights[action] >= self.ROWS:
            # Illegal Move:
            # Option A: Return huge penalty and end game (Simple)
            # Option B: Return huge penalty and ignore move (Safe)
            # We choose Option A to force Action Masking usage in the agent.
            return self._get_obs(), -1.0, True, False, {"error": "illegal_move"}
            
        # 2. Apply Gravity
        row = self.heights[action]
        self.board[row, action] = self.player_turn
        self.heights[action] += 1
        self.steps += 1
        
        # 3. Check Termination
        won = self._check_win(self.player_turn)
        draw = (self.steps == self.ROWS * self.COLS)
        
        terminated = won or draw
        truncated = False
        
        # 4. Reward Engineering (Sparse)
        reward = 0.0
        if won:
            reward = 1.0
        elif draw:
            reward = 0.0
            
        # 5. Flip Turn
        # If the game isn't over, we prepare the state for the NEXT player.
        # However, the reward returned is for the CURRENT player's action.
        if not terminated:
            self.player_turn *= -1
            
        info = {
            "turn": self.player_turn,
            "legal_moves": self._get_legal_moves_mask()
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Returns the CANONICAL observation.
        The network should always see 'Own Pieces' in Channel 0.
        """
        # Create masks
        p1_mask = (self.board == 1)
        p2_mask = (self.board == -1)
        
        if self.player_turn == 1:
            my_pieces = p1_mask
            opp_pieces = p2_mask
        else:
            # If it's Player -1's turn, we flip the board for the NN
            my_pieces = p2_mask
            opp_pieces = p1_mask
            
        # Channel 2: Broadcast valid moves to the whole grid 
        # (Helps CNNs understand column availability spatially)
        legal_mask_col = self._get_legal_moves_mask()
        legal_layer = np.tile(legal_mask_col, (self.ROWS, 1))
        
        obs = np.stack([my_pieces, opp_pieces, legal_layer]).astype(np.float32)
        return obs

    def _get_legal_moves_mask(self) -> np.ndarray:
        return (self.heights < self.ROWS).astype(np.int8)

    def _check_win(self, player_id: int) -> bool:
        """
        Checks if 'player_id' has won using convolution.
        Faster than Python loops for 4-in-a-row checks.
        """
        # Create a binary grid for the current player
        player_grid = (self.board == player_id).astype(np.int8)
        
        for kernel in self.kernels:
            # convolve2d calculates the sum of neighbors weighted by the kernel
            # If we have 4 ones in a row matching the kernel, the result will be 4.
            conv = convolve2d(player_grid, kernel, mode="valid")
            if (conv == 4).any():
                return True
        return False

    def render(self):
        if self.render_mode == "ansi" or self.render_mode == "human":
            print("\n 0 1 2 3 4 5 6")
            print("---------------")
            for r in range(self.ROWS - 1, -1, -1):
                row_str = "|"
                for c in range(self.COLS):
                    if self.board[r, c] == 1:
                        row_str += "X|" # Player 1
                    elif self.board[r, c] == -1:
                        row_str += "O|" # Player 2
                    else:
                        row_str += " |"
                print(row_str)
            print("---------------")
        
    def close(self):
        pass

# -------------------------------------------------------------------
# Helper for Parallel Training
# -------------------------------------------------------------------

def make_env(seed=None):
    """Factory function for creating environments in vectorized wrappers."""
    def _init():
        env = ConnectFourEnv(render_mode="rgb_array")
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

if __name__ == "__main__":
    # ---------------------------------------------------------------
    # DEMONSTRATION OF PARALLEL CAPABILITY
    # ---------------------------------------------------------------
    from gymnasium.vector import AsyncVectorEnv
    import time

    print("Checking Parallel Environment Performance...")
    
    # Create 4 environments running in separate processes
    num_envs = 4
    envs = AsyncVectorEnv([make_env(seed=i) for i in range(num_envs)])
    
    obs, infos = envs.reset()
    print(f"Observation Shape (Batched): {obs.shape}") # Should be (4, 3, 6, 7)
    
    start = time.time()
    # Run 1000 steps across 4 envs (4000 total steps)
    for _ in range(1000):
        # Random actions for all envs
        actions = envs.action_space.sample() 
        obs, rewards, term, trunc, infos = envs.step(actions)
    
    print(f"Finished 4000 steps in {time.time() - start:.2f} seconds.")
    envs.close()
    print("Environment is verified and ready for PPO.")