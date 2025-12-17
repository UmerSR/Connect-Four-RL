from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import copy

from envs.connect_four_env import ConnectFourEnv
from agents.simple_agents import RandomAgent, HeuristicAgent

try:
    import torch
    from training.manual_ppo_infer import ManualPPOLoader
except Exception:  # pragma: no cover - optional dependency
    torch = None
    ManualPPOLoader = None

try:
    from stable_baselines3 import DQN
except Exception:  # pragma: no cover - optional dependency
    DQN = None

try:
    import torch.nn as nn
    from torch.distributions.categorical import Categorical
except Exception:
    pass

class BaseOpponent(ABC):
    @abstractmethod
    def select_action(self, env: ConnectFourEnv) -> int:
        ...


class RandomOpponent(BaseOpponent):
    def __init__(self):
        self.agent = RandomAgent()

    def select_action(self, env: ConnectFourEnv) -> int:
        return self.agent.select_action(env)


class HeuristicOpponent(BaseOpponent):
    def __init__(self):
        self.agent = HeuristicAgent()

    def select_action(self, env: ConnectFourEnv) -> int:
        return self.agent.select_action(env)


class ManualPPOOpponent(BaseOpponent):
    def __init__(self, model_path: str, device: str = "cpu"):
        if ManualPPOLoader is None:
            raise ImportError("manual PPO loader not available; ensure torch is installed.")
        self.policy = ManualPPOLoader(model_path=model_path, device=device)

    def select_action(self, env: ConnectFourEnv) -> int:
        """
        Adapt canonical env obs (6,7,2) + mask into (6,7,3) expected by manual PPO:
            ch0: current player pieces
            ch1: opponent pieces
            ch2: legal mask broadcast
        """
        obs_dict = env._get_obs()
        planes = obs_dict["observation"]  # (6,7,2)
        cur_plane = planes[..., 0]
        opp_plane = planes[..., 1]
        # Flip vertically to match manual PPO training convention (row 0 as bottom)
        cur_plane = np.flipud(cur_plane)
        opp_plane = np.flipud(opp_plane)
        legal_mask = obs_dict["action_mask"].astype(bool)
        legal_layer = np.tile(legal_mask, (env.rows, 1))
        obs_hw3 = np.stack([cur_plane, opp_plane, legal_layer], axis=-1).astype(np.float32)
        action = self.policy.predict(obs_hw3, legal_mask)
        return int(action)


def _build_obs_hw3(env: ConnectFourEnv):
    """
    Convert the canonical dict observation to the 3xHxW tensor expected by DQN/PPO policies.
    Channels: current player, opponent, legal moves broadcast.
    """
    obs_dict = env._get_obs()
    planes = obs_dict["observation"]
    cur_plane = planes[..., 0]
    opp_plane = planes[..., 1]
    legal_mask = obs_dict["action_mask"].astype(np.float32)
    legal_layer = np.tile(legal_mask, (env.rows, 1))
    obs_chw = np.stack([cur_plane, opp_plane, legal_layer], axis=0).astype(np.float32)
    return obs_chw, legal_mask


class TorchDQNOpponent(BaseOpponent):
    """
    Torch-only DQN inference (no Stable-Baselines3 dependency) using the exported policy state_dict.
    Expects a .pth file containing q_net.* weights.
    """

    class _FeatureExtractor(torch.nn.Module):
        def __init__(self, rows: int = 6, cols: int = 7, features_dim: int = 512):
            super().__init__()
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.zeros(1, 3, rows, cols)
                n_flat = self.cnn(sample).shape[1]
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(n_flat, features_dim),
                torch.nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(x))

    class _QNet(torch.nn.Module):
        def __init__(self, rows: int = 6, cols: int = 7, features_dim: int = 512):
            super().__init__()
            self.features_extractor = TorchDQNOpponent._FeatureExtractor(rows, cols, features_dim)
            self.q_net = torch.nn.Sequential(
                torch.nn.Linear(features_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, cols),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.features_extractor(x)
            return self.q_net(feats)

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        if torch is None:
            raise ImportError("torch is required for DQN opponent.")
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"DQN model not found: {self.model_path}")
        self.policy = self._QNet().to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        # Only load q_net.* keys
        q_state = {k[len("q_net.") :]: v for k, v in state.items() if k.startswith("q_net.")}
        self.policy.load_state_dict(q_state, strict=True)
        self.policy.eval()

    def select_action(self, env: ConnectFourEnv) -> int:
        obs_chw, legal_mask = _build_obs_hw3(env)
        obs_t = torch.tensor(obs_chw).float().unsqueeze(0).to(self.device)
        mask_t = torch.tensor(legal_mask, device=self.device)
        with torch.no_grad():
            q_values = self.policy(obs_t)
            q_values = q_values.masked_fill(mask_t == 0, -1e9)
            action = torch.argmax(q_values, dim=1).item()
        return int(action)


class DQNOpponent(BaseOpponent):
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        model_path = Path(model_path)
        if model_path.suffix == ".pth" or DQN is None:
            # Torch-only path or SB3 not installed
            self.impl: BaseOpponent = TorchDQNOpponent(model_path=model_path, device=device)
        else:
            if DQN is None or torch is None:
                raise ImportError("stable-baselines3 DQN (and torch) are required for DQN opponent.")
            if not model_path.exists():
                raise FileNotFoundError(f"DQN model not found: {model_path}")
            self.model = DQN.load(model_path, device=device)
            self.model.policy.eval()
            self.impl = None
        self.model_path = model_path
        self.device = device

    def select_action(self, env: ConnectFourEnv) -> int:
        if getattr(self, "impl", None) is not None:
            return self.impl.select_action(env)
        obs_chw, legal_mask = _build_obs_hw3(env)
        obs_t = torch.tensor(obs_chw).float().unsqueeze(0).to(self.model.device)
        mask_t = torch.tensor(legal_mask, device=self.model.device)
        with torch.no_grad():
            q_values = self.model.policy.q_net(self.model.policy.features_extractor(obs_t))
            q_values = q_values.masked_fill(mask_t == 0, -1e9)
            action = torch.argmax(q_values, dim=1).item()
        return int(action)


class ReinforceOpponent(BaseOpponent):
    """
    Simple REINFORCE-style policy loaded from a saved state_dict.
    Assumes the same architecture used in the notebooks: 3xHxW input, conv stack -> 512 -> logits.
    """

    class _PolicyNet(nn.Module):
        def __init__(self, rows: int = 6, cols: int = 7, hidden_dim: int = 512):
            super().__init__()
            self.rows, self.cols = rows, cols
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, rows, cols)
                n_flat = self.cnn(dummy).shape[1]
            self.linear = nn.Sequential(
                nn.Linear(n_flat, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, cols),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(x))

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        if torch is None:
            raise ImportError("torch is required for REINFORCE opponent.")
        self.device = torch.device(device)
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"REINFORCE model not found: {model_path}")
        self.policy = self._PolicyNet().to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(state, strict=False)
        self.policy.eval()

    def select_action(self, env: ConnectFourEnv) -> int:
        obs_chw, legal_mask = _build_obs_hw3(env)
        obs_t = torch.tensor(obs_chw).float().unsqueeze(0).to(self.device)
        mask_t = torch.tensor(legal_mask, device=self.device)
        with torch.no_grad():
            logits = self.policy(obs_t)
            logits = logits.masked_fill(mask_t == 0, -1e9)
            action = torch.argmax(logits, dim=1).item()
        return int(action)
    
class GuidedOpponent(BaseOpponent):
    """
    A wrapper that performs 1-ply lookahead (Win/Block) before deferring to
    the underlying RL policy.
    """
    def __init__(self, underlying_opponent: BaseOpponent, env_sim: ConnectFourEnv):
        self.underlying_opponent = underlying_opponent
        
        # Create an internal, lightweight copy of the environment for simulation
        # This is CRUCIAL for the lookahead check.
        self.env_sim = env_sim
        self.name = f"GUIDED({underlying_opponent.name if hasattr(underlying_opponent, 'name') else underlying_opponent.__class__.__name__})"

    def _lookahead_check(self, env: ConnectFourEnv) -> Optional[int]:
        """
        Check for immediate Win (1) or Block (2) moves.
        Returns: Winning/Blocking column index (int) or None.
        """
        # 1. Sync internal simulation environment with the current game state
        self.env_sim.board = copy.deepcopy(env.board)
        self.env_sim.current_player = env.current_player
        
        # Get legal moves once
        legal_mask = self.env_sim._legal_mask()
        valid_cols = np.flatnonzero(legal_mask)
        
        current_piece = self.env_sim.current_player + 1
        opp_piece = 2 if current_piece == 1 else 1
        
        # --- 1. Check for immediate Win (My Turn) ---
        for col in valid_cols:
            row = self.env_sim._get_drop_row(col)
            
            # Simulate my move
            self.env_sim.board[row, col] = current_piece
            if self.env_sim._check_winner(current_piece):
                self.env_sim.board[row, col] = 0 # Undo
                return col # Immediate Win
            self.env_sim.board[row, col] = 0 # Undo

        # --- 2. Check for immediate Block (Opponent's Win) ---
        for col in valid_cols:
            row = self.env_sim._get_drop_row(col)
            
            # Simulate opponent's potential winning move
            self.env_sim.board[row, col] = opp_piece
            if self.env_sim._check_winner(opp_piece):
                self.env_sim.board[row, col] = 0 # Undo
                return col # Immediate Block
            self.env_sim.board[row, col] = 0 # Undo

        return None # No critical move found

    def select_action(self, env: ConnectFourEnv) -> int:
        """
        Decision Hierarchy: Win > Block > RL Policy.
        """
        
        # 1. Lookahead Check
        critical_action = self._lookahead_check(env)
        
        if critical_action is not None:
            return critical_action
        else:
            # 2. Defer to RL Policy
            return self.underlying_opponent.select_action(env)


def get_opponent(kind: str, model_path: Optional[str] = None, device: str = "cpu") -> BaseOpponent:
    # kind = kind.lower()
    
    kind_lower = kind.lower()
    
    # Check for the GUIDED prefix
    is_guided = kind_lower.startswith("guided_")
    if is_guided:
        # Extract the base kind (e.g., 'ppo' from 'guided_ppo')
        base_kind = kind_lower[7:] 
    else:
        base_kind = kind_lower
        
    if base_kind == "random":
        base_opponent = RandomOpponent()
    elif base_kind == "heuristic":
        base_opponent = HeuristicOpponent()
    elif base_kind in ("ppo", "ppo_pool", "ppo_dense", "ppo_connect4_8020_dense", "ppo_new"):
        if not model_path:
            raise ValueError("model_path required for PPO opponent")
        base_opponent = ManualPPOOpponent(model_path=model_path, device=device)
    elif base_kind == "dqn":
        if not model_path:
            raise ValueError("model_path required for DQN opponent")
        base_opponent = DQNOpponent(model_path=model_path, device=device)
    elif base_kind in ("reinforce", "reinforce_ts", "reinforce_manual", "reinforce_tianshou"):
        if not model_path:
            raise ValueError("model_path required for REINFORCE opponent")
        base_opponent = ReinforceOpponent(model_path=model_path, device=device)
    else:
        raise ValueError(f"Unknown opponent type: {base_kind}")


    if is_guided:
        # Create a fresh Env instance for the Guided Agent's lookahead sim
        env_sim = ConnectFourEnv()
        return GuidedOpponent(base_opponent, env_sim) 
        
    return base_opponent