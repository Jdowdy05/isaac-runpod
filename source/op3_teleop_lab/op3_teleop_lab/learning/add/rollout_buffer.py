from __future__ import annotations

import torch


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        diff_dim: int,
        device: torch.device,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs = torch.zeros((rollout_steps, num_envs, obs_dim), device=device, dtype=torch.float32)
        self.actions = torch.zeros((rollout_steps, num_envs, action_dim), device=device, dtype=torch.float32)
        self.log_probs = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float32)
        self.values = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float32)
        self.task_rewards = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float32)
        self.dones = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float32)
        self.diffs = torch.zeros((rollout_steps, num_envs, diff_dim), device=device, dtype=torch.float32)
        self.returns = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float32)
        self.advantages = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float32)
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        task_rewards: torch.Tensor,
        dones: torch.Tensor,
        diffs: torch.Tensor,
    ) -> None:
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.values[self.step] = values
        self.task_rewards[self.step] = task_rewards
        self.dones[self.step] = dones
        self.diffs[self.step] = diffs
        self.step += 1

    def compute_returns_and_advantages(
        self,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = torch.zeros_like(next_values)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_value = next_values
            else:
                next_value = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]

            delta = rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
        self.returns[:] = self.advantages + self.values

    def flattened(self) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs.reshape(-1, self.obs.shape[-1]),
            "actions": self.actions.reshape(-1, self.actions.shape[-1]),
            "log_probs": self.log_probs.reshape(-1),
            "values": self.values.reshape(-1),
            "returns": self.returns.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "diffs": self.diffs.reshape(-1, self.diffs.shape[-1]),
            "task_rewards": self.task_rewards.reshape(-1),
        }
