from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .config import ADDTrainingConfig, OptimizerConfig
from .networks import (
    DeterministicTeacherPolicy,
    DifferentialDiscriminator,
    TemporalStudentPolicy,
    ValueNetwork,
)
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
        critic_obs_dim: int | None = None,
    ) -> None:
        self.env = env
        self.actor_obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim if critic_obs_dim is not None else obs_dim
        self.action_dim = action_dim
        self.diff_dim = diff_dim
        self.cfg = config
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        history_steps = int(getattr(self.env.cfg, "actor_history_steps", 1))
        self.teacher_policy = DeterministicTeacherPolicy(
            obs_dim=self.critic_obs_dim,
            act_dim=action_dim,
            hidden_dims=config.teacher_hidden_dims,
            activation=config.activation,
            exploration_std=config.teacher_exploration_std,
        ).to(device)
        self.student_policy = TemporalStudentPolicy(
            obs_dim=obs_dim,
            act_dim=action_dim,
            history_steps=history_steps,
            rnn_hidden_dim=config.student_rnn_hidden_dim,
            hidden_dims=config.student_hidden_dims,
            activation=config.activation,
        ).to(device)
        self.value = ValueNetwork(
            obs_dim=self.critic_obs_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(device)
        self.discriminator = DifferentialDiscriminator(
            diff_dim=diff_dim,
            hidden_dims=config.disc_hidden_dims,
            activation=config.activation,
        ).to(device)

        self.teacher_optimizer = _make_optimizer(config.teacher_optimizer, self.teacher_policy.parameters())
        self.student_optimizer = _make_optimizer(config.student_optimizer, self.student_policy.parameters())
        self.critic_optimizer = _make_optimizer(config.critic_optimizer, self.value.parameters())
        self.disc_optimizer = _make_optimizer(config.disc_optimizer, self.discriminator.parameters())

        self.diff_normalizer = DiffNormalizer(diff_dim, device=device)
        self.replay_buffer = TensorReplayBuffer(config.disc_replay_capacity, diff_dim, device=device)

        self.rollout_buffer = RolloutBuffer(
            rollout_steps=config.rollout_steps,
            num_envs=env.num_envs,
            actor_obs_dim=self.actor_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            action_dim=action_dim,
            diff_dim=diff_dim,
            device=device,
        )
        self.running_task_returns = torch.zeros(env.num_envs, device=device)
        self.running_disc_returns = torch.zeros(env.num_envs, device=device)

        obs_dict, _ = self.env.reset()
        self.actor_obs = obs_dict["policy"]
        self.critic_obs = obs_dict.get("critic", self.actor_obs)
        self.iteration = 0

        # Backward-compatibility alias for scripts that still reference trainer.policy.
        self.policy = self.student_policy

    def train(self, num_iterations: int | None = None) -> None:
        max_iterations = num_iterations or self.cfg.max_iterations
        for iteration in range(1, max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()
            rollout_stats = self.collect_rollout()
            teacher_stats = self.update_teacher()
            student_stats = self.update_student()
            disc_stats = self.update_discriminator()
            elapsed = time.time() - iter_start

            if iteration % self.cfg.log_interval == 0 or iteration == 1:
                log_data = {
                    "iteration": iteration,
                    "elapsed_s": round(elapsed, 3),
                    **rollout_stats,
                    **teacher_stats,
                    **student_stats,
                    **disc_stats,
                }
                print(json.dumps(log_data, sort_keys=True))

            if iteration % self.cfg.save_interval == 0 or iteration == max_iterations:
                self.save(self.out_dir / f"add_op3_iter_{iteration:06d}.pt")

    def collect_rollout(self) -> dict[str, float]:
        self.rollout_buffer.step = 0
        completed_task_returns = []
        completed_disc_returns = []
        teacher_action_abs_means = []
        student_action_abs_means = []

        for _ in range(self.cfg.rollout_steps):
            with torch.no_grad():
                teacher_mean_actions = self.teacher_policy.deterministic(self.critic_obs)
                actions, log_probs = self.teacher_policy.sample(self.critic_obs)
                values = self.value(self.critic_obs)
                student_actions = self.student_policy.deterministic(self.actor_obs)

            next_obs, task_reward, terminated, truncated, extras = self.env.step(actions)
            next_actor_obs = next_obs["policy"]
            next_critic_obs = next_obs.get("critic", next_actor_obs)
            dones = (terminated | truncated).float()
            diffs = extras["add_diff"]

            teacher_action_abs_means.append(teacher_mean_actions.abs().mean().detach())
            student_action_abs_means.append(student_actions.abs().mean().detach())

            self.rollout_buffer.add(
                actor_obs=self.actor_obs,
                critic_obs=self.critic_obs,
                actions=actions,
                log_probs=log_probs,
                values=values,
                task_rewards=task_reward,
                dones=dones,
                diffs=diffs,
            )

            self.actor_obs = next_actor_obs
            self.critic_obs = next_critic_obs

        with torch.no_grad():
            next_values = self.value(self.critic_obs)

        flat_diffs = self.rollout_buffer.diffs.reshape(-1, self.diff_dim)
        self.diff_normalizer.record(flat_diffs)
        self.diff_normalizer.update()

        with torch.no_grad():
            disc_rewards = self.compute_disc_rewards(flat_diffs).view(self.cfg.rollout_steps, self.env.num_envs)

        rewards = self.cfg.task_reward_weight * self.rollout_buffer.task_rewards + self.cfg.disc_reward_weight * disc_rewards
        self.rollout_buffer.compute_returns_and_advantages(
            rewards=rewards,
            next_values=next_values,
            gamma=self.cfg.discount,
            gae_lambda=self.cfg.gae_lambda,
        )

        for step in range(self.cfg.rollout_steps):
            self.running_task_returns += self.rollout_buffer.task_rewards[step]
            self.running_disc_returns += disc_rewards[step]
            done = self.rollout_buffer.dones[step] > 0
            if torch.any(done):
                completed_task_returns.extend(self.running_task_returns[done].detach().cpu().tolist())
                completed_disc_returns.extend(self.running_disc_returns[done].detach().cpu().tolist())
                self.running_task_returns[done] = 0.0
                self.running_disc_returns[done] = 0.0

        return {
            "task_reward_mean": float(self.rollout_buffer.task_rewards.mean().item()),
            "disc_reward_mean": float(disc_rewards.mean().item()),
            "total_reward_mean": float(rewards.mean().item()),
            "episode_task_return_mean": float(sum(completed_task_returns) / max(1, len(completed_task_returns))),
            "episode_disc_return_mean": float(sum(completed_disc_returns) / max(1, len(completed_disc_returns))),
            "teacher_action_abs_mean": float(torch.stack(teacher_action_abs_means).mean().item()),
            "student_action_abs_mean": float(torch.stack(student_action_abs_means).mean().item()),
        }

    def update_teacher(self) -> dict[str, float]:
        batch = self.rollout_buffer.flattened()
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / torch.clamp(advantages.std(), min=1.0e-6)
        batch["advantages"] = advantages
        num_samples = batch["critic_obs"].shape[0]
        minibatch_size = min(self.cfg.minibatch_size, num_samples)

        teacher_losses = []
        critic_losses = []
        entropies = []

        num_epochs = max(self.cfg.teacher_epochs, self.cfg.critic_epochs)
        for epoch in range(num_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                critic_obs = batch["critic_obs"][idx]
                actions = batch["actions"][idx]
                old_log_probs = batch["log_probs"][idx]
                adv = batch["advantages"][idx]
                returns = batch["returns"][idx]

                log_probs, entropy = self.teacher_policy.evaluate_actions(critic_obs, actions)
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip_ratio, 1.0 + self.cfg.ppo_clip_ratio) * adv
                teacher_loss = -torch.min(unclipped, clipped).mean() - self.cfg.entropy_coef * entropy.mean()

                values = self.value(critic_obs)
                critic_loss = 0.5 * self.cfg.value_loss_coef * (returns - values).square().mean()

                if epoch < self.cfg.teacher_epochs:
                    self.teacher_optimizer.zero_grad(set_to_none=True)
                    teacher_loss.backward()
                    nn.utils.clip_grad_norm_(self.teacher_policy.parameters(), self.cfg.max_grad_norm)
                    self.teacher_optimizer.step()

                if epoch < self.cfg.critic_epochs:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.max_grad_norm)
                    self.critic_optimizer.step()

                if epoch < self.cfg.teacher_epochs:
                    teacher_losses.append(teacher_loss.detach())
                    entropies.append(entropy.mean().detach())
                if epoch < self.cfg.critic_epochs:
                    critic_losses.append(critic_loss.detach())

        return {
            "teacher_loss": float(torch.stack(teacher_losses).mean().item()) if teacher_losses else 0.0,
            "critic_loss": float(torch.stack(critic_losses).mean().item()) if critic_losses else 0.0,
            "teacher_entropy": float(torch.stack(entropies).mean().item()) if entropies else 0.0,
        }

    def update_student(self) -> dict[str, float]:
        batch = self.rollout_buffer.flattened()
        num_samples = batch["actor_obs"].shape[0]
        minibatch_size = min(self.cfg.student_batch_size, num_samples)
        losses = []

        for _ in range(self.cfg.student_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                actor_obs = batch["actor_obs"][idx]
                critic_obs = batch["critic_obs"][idx]

                with torch.no_grad():
                    teacher_targets = self.teacher_policy.deterministic(critic_obs)

                student_actions = self.student_policy.deterministic(actor_obs)
                bc_loss = self.cfg.student_bc_weight * (student_actions - teacher_targets).square().mean()

                self.student_optimizer.zero_grad(set_to_none=True)
                bc_loss.backward()
                nn.utils.clip_grad_norm_(self.student_policy.parameters(), self.cfg.max_grad_norm)
                self.student_optimizer.step()
                losses.append(bc_loss.detach())

        return {
            "student_bc_loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
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

    def deployment_actions(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.student_policy.deterministic(actor_obs)

    def teacher_actions(
        self,
        critic_obs: torch.Tensor,
        sample: bool = False,
    ) -> torch.Tensor:
        if sample:
            actions, _ = self.teacher_policy.sample(critic_obs)
            return actions
        return self.teacher_policy.deterministic(critic_obs)

    def save(self, checkpoint_path: str | Path) -> None:
        payload = {
            "iteration": self.iteration,
            "config": self.cfg.__dict__,
            "teacher_policy": self.teacher_policy.state_dict(),
            "student_policy": self.student_policy.state_dict(),
            "value": self.value.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "teacher_optimizer": self.teacher_optimizer.state_dict(),
            "student_optimizer": self.student_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "diff_normalizer": self.diff_normalizer.state_dict(),
        }
        torch.save(payload, checkpoint_path)

    def load(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.iteration = int(payload.get("iteration", 0))

        if "teacher_policy" in payload:
            self.teacher_policy.load_state_dict(payload["teacher_policy"])
        elif "policy" in payload:
            self.teacher_policy.load_state_dict(payload["policy"])

        if "student_policy" in payload:
            self.student_policy.load_state_dict(payload["student_policy"])

        self.value.load_state_dict(payload["value"])
        self.discriminator.load_state_dict(payload["discriminator"])

        if "teacher_optimizer" in payload:
            self.teacher_optimizer.load_state_dict(payload["teacher_optimizer"])
        elif "actor_optimizer" in payload:
            self.teacher_optimizer.load_state_dict(payload["actor_optimizer"])
        if "student_optimizer" in payload:
            self.student_optimizer.load_state_dict(payload["student_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
        self.diff_normalizer.load_state_dict(payload["diff_normalizer"])
