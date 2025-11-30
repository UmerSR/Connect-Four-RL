from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from envs.connect_four_env import ConnectFourEnv
from agents.simple_agents import RandomAgent, HeuristicAgent

try:
    import torch
    from training.manual_ppo_infer import ManualPPOLoader
except Exception:  # pragma: no cover - optional dependency
    torch = None
    ManualPPOLoader = None


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


def get_opponent(kind: str, model_path: Optional[str] = None, device: str = "cpu") -> BaseOpponent:
    kind = kind.lower()
    if kind == "random":
        return RandomOpponent()
    if kind == "heuristic":
        return HeuristicOpponent()
    if kind == "ppo":
        if not model_path:
            raise ValueError("model_path required for PPO opponent")
        return ManualPPOOpponent(model_path=model_path, device=device)
    raise ValueError(f"Unknown opponent type: {kind}")
