"""
Minimal PPO training scaffold for ConnectFourEnv self-play.
This is intentionally lightweight so it can be extended for experiments.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from envs.connect_four_env import ConnectFourEnv


@dataclass
class Transition:
    obs: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(self, rows: int, cols: int, hidden_dim: int = 128):
        super().__init__()
        self.rows = rows
        self.cols = cols

        # Observation shape: (B, rows, cols, 2) -> convert to (B, 2, rows, cols)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out = rows * cols * 64
        self.mlp = nn.Sequential(
            nn.Linear(conv_out, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, cols)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        # obs: (B, rows, cols, 2)
        x = obs.permute(0, 3, 1, 2)  # to (B, C, H, W)
        x = self.encoder(x)
        x = self.mlp(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        logits, value = self.forward(obs)
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value


def prepare_tensors(obs_dict, device):
    obs = torch.from_numpy(obs_dict["observation"]).float().unsqueeze(0).to(device)
    mask = torch.from_numpy(obs_dict["action_mask"]).float().unsqueeze(0).to(device)
    return obs, mask


def collect_rollout(
    env: ConnectFourEnv,
    policy: ActorCritic,
    device: torch.device,
    rollout_steps: int,
) -> List[Transition]:
    transitions: List[Transition] = []
    obs_dict, _ = env.reset()
    for _ in range(rollout_steps):
        obs_t, mask_t = prepare_tensors(obs_dict, device)
        with torch.no_grad():
            action_t, logprob_t, value_t = policy.act(obs_t, mask_t)

        next_obs, reward, terminated, truncated, _ = env.step(int(action_t.item()))
        done = terminated or truncated

        transitions.append(
            Transition(
                obs=obs_t.squeeze(0),
                mask=mask_t.squeeze(0),
                action=action_t.detach(),
                logprob=logprob_t.detach(),
                value=value_t.detach(),
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=torch.tensor(done, dtype=torch.float32, device=device),
            )
        )

        obs_dict = next_obs
        if done:
            obs_dict, _ = env.reset()
    return transitions


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_update(
    policy: ActorCritic,
    optimizer: optim.Optimizer,
    transitions: List[Transition],
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    train_epochs: int = 4,
    batch_size: int = 64,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    device = next(policy.parameters()).device
    obs = torch.stack([t.obs for t in transitions]).to(device)
    mask = torch.stack([t.mask for t in transitions]).to(device)
    actions = torch.stack([t.action for t in transitions]).to(device)
    old_logprobs = torch.stack([t.logprob for t in transitions]).to(device)
    rewards = torch.stack([t.reward for t in transitions]).to(device)
    dones = torch.stack([t.done for t in transitions]).to(device)

    with torch.no_grad():
        _, values = policy.forward(obs)
    values = torch.cat([values, torch.tensor([0.0], device=device)])

    advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = len(transitions)
    for _ in range(train_epochs):
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_obs = obs[batch_idx]
            batch_mask = mask[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_logprobs = old_logprobs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            logits, value = policy.forward(batch_obs)
            masked_logits = logits.masked_fill(batch_mask == 0, -1e9)
            dist = Categorical(logits=masked_logits)
            logprobs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = (logprobs - batch_old_logprobs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(value, batch_returns)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()


def train(
    total_steps: int = 5000,
    rollout_steps: int = 256,
    learning_rate: float = 3e-4,
    device: str | torch.device = "cpu",
):
    device = torch.device(device)
    env = ConnectFourEnv()
    policy = ActorCritic(env.rows, env.cols).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    steps_collected = 0
    while steps_collected < total_steps:
        transitions = collect_rollout(env, policy, device, rollout_steps)
        steps_collected += len(transitions)
        ppo_update(policy, optimizer, transitions)
        print(f"Trained on {steps_collected}/{total_steps} steps")

    return policy


if __name__ == "__main__":
    train()
