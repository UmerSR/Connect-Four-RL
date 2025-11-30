"""
Manual PPO inference wrapper for the pre-trained model in ppo_implementation/ppo_connect4_8020.pth.

Observation expected: (3, 6, 7) float32
    ch0: current player's pieces
    ch1: opponent's pieces
    ch2: legal move mask broadcast across rows
Action space: Discrete(7)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOPolicyCNN(nn.Module):
    def __init__(self, rows: int = 6, cols: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, rows, cols)
            conv_out = self.conv(dummy).view(1, -1).shape[1]
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cols),
        )

    def forward(self, x):
        # x: (B, 3, rows, cols)
        x = self.conv(x)
        logits = self.policy_head(x)
        return logits


class ManualPPOLoader:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.policy = PPOPolicyCNN().to(self.device)
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("policy."):
                nk = nk[len("policy.") :]
            new_state[nk] = v
        self.policy.load_state_dict(new_state, strict=False)
        self.policy.eval()

    @torch.no_grad()
    def predict(self, observation_hw3, legal_mask, deterministic: bool = True) -> int:
        """
        observation_hw3: np.ndarray shape (rows, cols, 3) or (3, rows, cols)
        legal_mask: np.ndarray shape (7,) bool
        """
        import numpy as np

        obs = observation_hw3
        if obs.shape == (6, 7, 3):
            obs = np.transpose(obs, (2, 0, 1))
        elif obs.shape == (6, 7, 2):
            legal_layer = np.tile(legal_mask, (6, 1))
            obs = np.transpose(np.stack([obs[..., 0], obs[..., 1], legal_layer], axis=-1), (2, 0, 1))
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # (1,3,H,W)
        logits = self.policy(obs_t).squeeze(0)  # (7,)
        logits = logits.masked_fill(torch.tensor(~legal_mask, device=self.device), -1e9)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            action = torch.multinomial(probs, 1).item()
        return action
