from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .config import ADDTrainingConfig, OptimizerConfig
from .networks import DifferentialDiscriminator, GaussianPolicy, ValueNetwork
from .normalizers import DiffNormalizer
from .replay_buffer import TensorReplayBuffer
from .rollout_buffer import RolloutBuffer


def _make_optimizer(config: OptimizerConfig, params) -> torch.optim.Optimizer:
    optimizer_type = config.type.lower()
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer type: {config.type}")


class ADDTrainer:
    def __init__(
        self,
        env,
        obs_dim: int,
        action_dim: int,
        diff_dim: int,
        config: ADDTrainingConfig,
        device: torch.device,
        out_dir: str | Path,
    ) -> None:
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.diff_dim = diff_dim
        self.cfg = config
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.policy = GaussianPolicy(
            obs_dim=obs_dim,
            act_dim=action_dim,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            fixed_action_std=config.fixed_action_std,
        ).to(device)
        self.value = ValueNetwork(
            obs_dim=obs_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(device)
        self.discriminator = DifferentialDiscriminator(
            diff_dim=diff_dim,
            hidden_dims=config.disc_hidden_dims,
            activation=config.activation,
        ).to(device)

        self.actor_optimizer = _make_optimizer(config.actor_optimizer, self.policy.parameters())
        self.critic_optimizer = _make_optimizer(config.critic_optimizer, self.value.parameters())
        self.disc_optimizer = _make_optimizer(config.disc_optimizer, self.discriminator.parameters())

        self.diff_normalizer = DiffNormalizer(diff_dim, device=device)
        self.replay_buffer = TensorReplayBuffer(config.disc_replay_capacity, diff_dim, device=device)

        self.rollout_buffer = RolloutBuffer(
            rollout_steps=config.rollout_steps,
            num_envs=env.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            diff_dim=diff_dim,
            device=device,
        )

        self.obs, _ = self.env.reset()
        self.obs = self.obs["policy"]
        self.iteration = 0

    def train(self, num_iterations: int | None = None) -> None:
        max_iterations = num_iterations or self.cfg.max_iterations
        for iteration in range(1, max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()
            rollout_stats = self.collect_rollout()
            ppo_stats = self.update_policy()
            disc_stats = self.update_discriminator()
            elapsed = time.time() - iter_start

            if iteration % self.cfg.log_interval == 0 or iteration == 1:
                log_data = {
                    "iteration": iteration,
                    "elapsed_s": round(elapsed, 3),
                    **rollout_stats,
                    **ppo_stats,
                    **disc_stats,
                }
                print(json.dumps(log_data, sort_keys=True))

            if iteration % self.cfg.save_interval == 0 or iteration == max_iterations:
                self.save(self.out_dir / f"add_op3_iter_{iteration:06d}.pt")

    def collect_rollout(self) -> dict[str, float]:
        self.rollout_buffer.step = 0
        completed_task_returns = []
        completed_disc_returns = []
        running_task_returns = torch.zeros(self.env.num_envs, device=self.device)
        running_disc_returns = torch.zeros(self.env.num_envs, device=self.device)

        for _ in range(self.cfg.rollout_steps):
            with torch.no_grad():
                actions, log_probs = self.policy.sample(self.obs)
                values = self.value(self.obs)

            next_obs, task_reward, terminated, truncated, extras = self.env.step(actions)
            next_obs = next_obs["policy"]
            dones = (terminated | truncated).float()
            diffs = extras["add_diff"]

            self.rollout_buffer.add(
                obs=self.obs,
                actions=actions,
                log_probs=log_probs,
                values=values,
                task_rewards=task_reward,
                dones=dones,
                diffs=diffs,
            )

            self.obs = next_obs

        with torch.no_grad():
            next_values = self.value(self.obs)

        flat_diffs = self.rollout_buffer.diffs.reshape(-1, self.diff_dim)
        self.diff_normalizer.record(flat_diffs)
        self.diff_normalizer.update()

        with torch.no_grad():
            disc_rewards = self.compute_disc_rewards(flat_diffs).view(self.cfg.rollout_steps, self.env.num_envs)

        rewards = (
            self.cfg.task_reward_weight * self.rollout_buffer.task_rewards
            + self.cfg.disc_reward_weight * disc_rewards
        )
        self.rollout_buffer.compute_returns_and_advantages(
            rewards=rewards,
            next_values=next_values,
            gamma=self.cfg.discount,
            gae_lambda=self.cfg.gae_lambda,
        )

        running_task_returns += torch.sum(self.rollout_buffer.task_rewards, dim=0)
        running_disc_returns += torch.sum(disc_rewards, dim=0)
        last_done = self.rollout_buffer.dones[-1] > 0
        if torch.any(last_done):
            completed_task_returns.extend(running_task_returns[last_done].detach().cpu().tolist())
            completed_disc_returns.extend(running_disc_returns[last_done].detach().cpu().tolist())

        return {
            "task_reward_mean": float(self.rollout_buffer.task_rewards.mean().item()),
            "disc_reward_mean": float(disc_rewards.mean().item()),
            "total_reward_mean": float(rewards.mean().item()),
            "episode_task_return_mean": float(sum(completed_task_returns) / max(1, len(completed_task_returns))),
            "episode_disc_return_mean": float(sum(completed_disc_returns) / max(1, len(completed_disc_returns))),
        }

    def update_policy(self) -> dict[str, float]:
        batch = self.rollout_buffer.flattened()
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / torch.clamp(advantages.std(), min=1.0e-6)
        batch["advantages"] = advantages
        num_samples = batch["obs"].shape[0]
        minibatch_size = min(self.cfg.minibatch_size, num_samples)

        actor_losses = []
        critic_losses = []
        entropies = []

        num_epochs = max(self.cfg.actor_epochs, self.cfg.critic_epochs)
        for epoch in range(num_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                obs = batch["obs"][idx]
                actions = batch["actions"][idx]
                old_log_probs = batch["log_probs"][idx]
                adv = batch["advantages"][idx]
                returns = batch["returns"][idx]

                log_probs, entropy = self.policy.evaluate_actions(obs, actions)
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip_ratio, 1.0 + self.cfg.ppo_clip_ratio) * adv
                actor_loss = -torch.min(unclipped, clipped).mean() - self.cfg.entropy_coef * entropy.mean()

                values = self.value(obs)
                critic_loss = 0.5 * self.cfg.value_loss_coef * (returns - values).square().mean()

                if epoch < self.cfg.actor_epochs:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                    self.actor_optimizer.step()

                if epoch < self.cfg.critic_epochs:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.max_grad_norm)
                    self.critic_optimizer.step()

                if epoch < self.cfg.actor_epochs:
                    actor_losses.append(actor_loss.detach())
                    entropies.append(entropy.mean().detach())
                if epoch < self.cfg.critic_epochs:
                    critic_losses.append(critic_loss.detach())

        return {
            "actor_loss": float(torch.stack(actor_losses).mean().item()) if actor_losses else 0.0,
            "critic_loss": float(torch.stack(critic_losses).mean().item()) if critic_losses else 0.0,
            "policy_entropy": float(torch.stack(entropies).mean().item()) if entropies else 0.0,
        }

    def update_discriminator(self) -> dict[str, float]:
        flat_diffs = self.rollout_buffer.diffs.reshape(-1, self.diff_dim)
        self.replay_buffer.add(flat_diffs)

        num_samples = flat_diffs.shape[0]
        minibatch_size = min(self.cfg.minibatch_size, num_samples)
        steps_per_epoch = math.ceil(num_samples / minibatch_size)

        losses = []
        grad_penalties = []
        pos_accs = []
        neg_accs = []

        for _ in range(self.cfg.disc_epochs * steps_per_epoch):
            idx = torch.randint(0, num_samples, (minibatch_size,), device=self.device)
            current_diff = flat_diffs[idx]
            replay_count = min(current_diff.shape[0], self.cfg.disc_replay_samples, self.replay_buffer.size)
            if replay_count > 0:
                replay_diff = self.replay_buffer.sample(replay_count)
                neg_diff = torch.cat((current_diff, replay_diff), dim=0)
            else:
                neg_diff = current_diff

            norm_neg_diff = self.diff_normalizer.normalize(neg_diff.detach()).clone().requires_grad_(True)
            pos_diff = torch.zeros((current_diff.shape[0], self.diff_dim), device=self.device, dtype=torch.float32)
            pos_diff.requires_grad_(True)

            pos_logits = self.discriminator(pos_diff)
            neg_logits = self.discriminator(norm_neg_diff)

            bce = nn.BCEWithLogitsLoss()
            pos_loss = bce(pos_logits, torch.ones_like(pos_logits))
            neg_loss = bce(neg_logits, torch.zeros_like(neg_logits))
            disc_loss = 0.5 * (pos_loss + neg_loss)

            logit_reg = torch.sum(self.discriminator.get_logit_weights().square())
            disc_loss = disc_loss + self.cfg.disc_logit_reg * logit_reg

            neg_grad = torch.autograd.grad(
                neg_logits,
                norm_neg_diff,
                grad_outputs=torch.ones_like(neg_logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            pos_grad = torch.autograd.grad(
                pos_logits,
                pos_diff,
                grad_outputs=torch.ones_like(pos_logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_penalty = 0.5 * (neg_grad.square().sum(dim=-1).mean() + pos_grad.square().sum(dim=-1).mean())
            disc_loss = disc_loss + self.cfg.disc_grad_penalty * grad_penalty

            self.disc_optimizer.zero_grad(set_to_none=True)
            disc_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.max_grad_norm)
            self.disc_optimizer.step()

            losses.append(disc_loss.detach())
            grad_penalties.append(grad_penalty.detach())
            pos_accs.append((pos_logits > 0.0).float().mean().detach())
            neg_accs.append((neg_logits < 0.0).float().mean().detach())

        return {
            "disc_loss": float(torch.stack(losses).mean().item()),
            "disc_grad_penalty": float(torch.stack(grad_penalties).mean().item()),
            "disc_pos_acc": float(torch.stack(pos_accs).mean().item()),
            "disc_neg_acc": float(torch.stack(neg_accs).mean().item()),
        }

    def compute_disc_rewards(self, flat_diffs: torch.Tensor) -> torch.Tensor:
        norm_diffs = self.diff_normalizer.normalize(flat_diffs)
        logits = self.discriminator(norm_diffs)
        prob = torch.sigmoid(logits)
        rewards = -torch.log(torch.clamp(1.0 - prob, min=1.0e-4))
        return rewards * self.cfg.disc_reward_scale

    def save(self, checkpoint_path: str | Path) -> None:
        payload = {
            "iteration": self.iteration,
            "config": self.cfg.__dict__,
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "diff_normalizer": self.diff_normalizer.state_dict(),
        }
        torch.save(payload, checkpoint_path)

    def load(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.iteration = int(payload.get("iteration", 0))
        self.policy.load_state_dict(payload["policy"])
        self.value.load_state_dict(payload["value"])
        self.discriminator.load_state_dict(payload["discriminator"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
        self.diff_normalizer.load_state_dict(payload["diff_normalizer"])
