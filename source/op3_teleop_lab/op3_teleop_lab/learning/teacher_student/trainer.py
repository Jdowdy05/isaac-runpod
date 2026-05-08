from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from op3_teleop_lab.learning.add.networks import DeterministicTeacherPolicy, TemporalStudentPolicy, ValueNetwork

from .config import OptimizerConfig, TeacherStudentTrainingConfig
from .rollout_buffer import TeacherStudentRolloutBuffer


def _make_optimizer(config: OptimizerConfig, params) -> torch.optim.Optimizer:
    optimizer_type = config.type.lower()
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer type: {config.type}")


class TeacherStudentTrainer:
    def __init__(
        self,
        env,
        obs_dim: int,
        action_dim: int,
        config: TeacherStudentTrainingConfig,
        device: torch.device,
        out_dir: str | Path,
        critic_obs_dim: int | None = None,
    ) -> None:
        self.env = env
        self.actor_obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim if critic_obs_dim is not None else obs_dim
        self.teacher_obs_dim = self.critic_obs_dim if config.teacher_uses_critic_obs else self.actor_obs_dim
        self.action_dim = action_dim
        self.cfg = config
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        history_steps = int(getattr(self.env.cfg, "actor_history_steps", 1))
        self.teacher_policy = DeterministicTeacherPolicy(
            obs_dim=self.teacher_obs_dim,
            act_dim=action_dim,
            hidden_dims=config.teacher_hidden_dims,
            activation=config.activation,
            exploration_std=config.teacher_exploration_std,
            output_init_scale=config.teacher_output_init_scale,
        ).to(device)
        self.student_policy = TemporalStudentPolicy(
            obs_dim=obs_dim,
            act_dim=action_dim,
            history_steps=history_steps,
            rnn_hidden_dim=config.student_rnn_hidden_dim,
            hidden_dims=config.student_hidden_dims,
            activation=config.activation,
            output_init_scale=config.student_output_init_scale,
        ).to(device)
        self.value = ValueNetwork(
            obs_dim=self.critic_obs_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(device)

        self.teacher_optimizer = _make_optimizer(config.teacher_optimizer, self.teacher_policy.parameters())
        self.student_optimizer = _make_optimizer(config.student_optimizer, self.student_policy.parameters())
        self.critic_optimizer = _make_optimizer(config.critic_optimizer, self.value.parameters())

        self.rollout_buffer = TeacherStudentRolloutBuffer(
            rollout_steps=config.rollout_steps,
            num_envs=env.num_envs,
            actor_obs_dim=self.actor_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            action_dim=action_dim,
            device=device,
        )
        self.running_task_returns = torch.zeros(env.num_envs, device=device)

        obs_dict, _ = self.env.reset()
        self.actor_obs = obs_dict["policy"]
        self.critic_obs = obs_dict.get("critic", self.actor_obs)
        self.teacher_iteration = 0
        self.student_iteration = 0
        self._checkpoint_thread: threading.Thread | None = None
        self._checkpoint_error: BaseException | None = None
        self._checkpoint_join_timeout_s = 120.0

        self.policy = self.student_policy

    def reset_env_state(self) -> None:
        obs_dict, _ = self.env.reset()
        self.actor_obs = obs_dict["policy"]
        self.critic_obs = obs_dict.get("critic", self.actor_obs)
        self.running_task_returns.zero_()

    def _select_teacher_obs(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> torch.Tensor:
        if self.cfg.teacher_uses_critic_obs:
            if critic_obs is None:
                raise ValueError("critic_obs is required when teacher_uses_critic_obs is enabled.")
            return critic_obs
        return actor_obs

    def _teacher_exploration_std_for_iteration(self, iteration: int) -> float:
        initial_std = float(self.cfg.teacher_exploration_std)
        final_std = float(self.cfg.teacher_exploration_final_std)
        decay_iterations = int(self.cfg.teacher_exploration_decay_iterations)
        if decay_iterations <= 0:
            return initial_std
        progress = min(1.0, max(0.0, (iteration - 1) / decay_iterations))
        return initial_std + progress * (final_std - initial_std)

    def train_teacher(self, num_iterations: int | None = None) -> None:
        max_iterations = num_iterations or self.cfg.teacher_max_iterations
        self.reset_env_state()
        try:
            for iteration in range(self.teacher_iteration + 1, max_iterations + 1):
                self.teacher_iteration = iteration
                teacher_exploration_std = self._teacher_exploration_std_for_iteration(iteration)
                self.teacher_policy.set_exploration_std(teacher_exploration_std)
                iter_start = time.time()
                rollout_stats = self.collect_teacher_rollout()
                update_stats = self.update_teacher()
                elapsed = time.time() - iter_start

                if iteration % self.cfg.log_interval == 0 or iteration == 1:
                    log_data = {
                        "stage": "teacher",
                        "iteration": iteration,
                        "elapsed_s": round(elapsed, 3),
                        "teacher_exploration_std": teacher_exploration_std,
                        **rollout_stats,
                        **update_stats,
                    }
                    print(json.dumps(log_data, sort_keys=True), flush=True)

                if iteration % self.cfg.save_interval == 0 or iteration == max_iterations:
                    self.save(self.out_dir / f"teacher_iter_{iteration:06d}.pt", stage="teacher")
        finally:
            self.wait_for_pending_checkpoint()

    def distill_student(self, num_iterations: int | None = None) -> None:
        max_iterations = num_iterations or self.cfg.student_max_iterations
        self.reset_env_state()
        try:
            for iteration in range(self.student_iteration + 1, max_iterations + 1):
                self.student_iteration = iteration
                iter_start = time.time()
                rollout_stats = self.collect_student_rollout()
                update_stats = self.update_student()
                elapsed = time.time() - iter_start

                if iteration % self.cfg.log_interval == 0 or iteration == 1:
                    log_data = {
                        "stage": "student",
                        "iteration": iteration,
                        "elapsed_s": round(elapsed, 3),
                        **rollout_stats,
                        **update_stats,
                    }
                    print(json.dumps(log_data, sort_keys=True), flush=True)

                if iteration % self.cfg.save_interval == 0 or iteration == max_iterations:
                    self.save(self.out_dir / f"student_iter_{iteration:06d}.pt", stage="student")
        finally:
            self.wait_for_pending_checkpoint()

    def _reset_rollout_state(self) -> None:
        self.rollout_buffer.reset()

    def _accumulate_episode_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        completed_task_returns: list[float],
    ) -> None:
        for step in range(self.cfg.rollout_steps):
            self.running_task_returns += rewards[step]
            done = dones[step] > 0
            if torch.any(done):
                completed_task_returns.extend(self.running_task_returns[done].detach().cpu().tolist())
                self.running_task_returns[done] = 0.0

    def collect_teacher_rollout(self) -> dict[str, float]:
        self._reset_rollout_state()
        completed_task_returns: list[float] = []
        teacher_action_abs_means = []
        sampled_action_abs_means = []
        sampled_action_abs_maxes = []

        for _ in range(self.cfg.rollout_steps):
            with torch.no_grad():
                teacher_obs = self._select_teacher_obs(self.actor_obs, self.critic_obs)
                teacher_mean_actions = self.teacher_policy.deterministic(teacher_obs)
                actions, log_probs = self.teacher_policy.sample(teacher_obs)
                values = self.value(self.critic_obs)

            next_obs, task_reward, terminated, truncated, _extras = self.env.step(actions)
            next_actor_obs = next_obs["policy"]
            next_critic_obs = next_obs.get("critic", next_actor_obs)
            dones = (terminated | truncated).float()

            teacher_action_abs_means.append(teacher_mean_actions.abs().mean().detach())
            sampled_action_abs = actions.abs()
            sampled_action_abs_means.append(sampled_action_abs.mean().detach())
            sampled_action_abs_maxes.append(sampled_action_abs.max().detach())

            self.rollout_buffer.add(
                actor_obs=self.actor_obs,
                critic_obs=self.critic_obs,
                actions=actions,
                log_probs=log_probs,
                values=values,
                task_rewards=task_reward,
                dones=dones,
            )

            self.actor_obs = next_actor_obs
            self.critic_obs = next_critic_obs

        with torch.no_grad():
            next_values = self.value(self.critic_obs)

        self.rollout_buffer.compute_returns_and_advantages(
            next_values=next_values,
            gamma=self.cfg.discount,
            gae_lambda=self.cfg.gae_lambda,
        )
        self._accumulate_episode_returns(self.rollout_buffer.task_rewards, self.rollout_buffer.dones, completed_task_returns)

        return {
            "task_reward_mean": float(self.rollout_buffer.task_rewards.mean().item()),
            "episode_task_return_mean": float(sum(completed_task_returns) / max(1, len(completed_task_returns))),
            "teacher_action_abs_mean": float(torch.stack(teacher_action_abs_means).mean().item()),
            "sampled_action_abs_mean": float(torch.stack(sampled_action_abs_means).mean().item()),
            "sampled_action_abs_max": float(torch.stack(sampled_action_abs_maxes).max().item()),
        }

    def collect_student_rollout(self) -> dict[str, float]:
        self._reset_rollout_state()
        completed_task_returns: list[float] = []
        teacher_action_abs_means = []
        student_action_abs_means = []

        for _ in range(self.cfg.rollout_steps):
            with torch.no_grad():
                teacher_obs = self._select_teacher_obs(self.actor_obs, self.critic_obs)
                teacher_actions = self.teacher_policy.deterministic(teacher_obs)
                student_actions = self.student_policy.deterministic(self.actor_obs)
                zero_log_probs = torch.zeros((self.env.num_envs,), device=self.device, dtype=torch.float32)
                zero_values = torch.zeros((self.env.num_envs,), device=self.device, dtype=torch.float32)

            next_obs, task_reward, terminated, truncated, _extras = self.env.step(teacher_actions)
            next_actor_obs = next_obs["policy"]
            next_critic_obs = next_obs.get("critic", next_actor_obs)
            dones = (terminated | truncated).float()

            teacher_action_abs_means.append(teacher_actions.abs().mean().detach())
            student_action_abs_means.append(student_actions.abs().mean().detach())

            self.rollout_buffer.add(
                actor_obs=self.actor_obs,
                critic_obs=self.critic_obs,
                actions=teacher_actions,
                log_probs=zero_log_probs,
                values=zero_values,
                task_rewards=task_reward,
                dones=dones,
            )

            self.actor_obs = next_actor_obs
            self.critic_obs = next_critic_obs

        self._accumulate_episode_returns(self.rollout_buffer.task_rewards, self.rollout_buffer.dones, completed_task_returns)

        return {
            "task_reward_mean": float(self.rollout_buffer.task_rewards.mean().item()),
            "episode_task_return_mean": float(sum(completed_task_returns) / max(1, len(completed_task_returns))),
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
                actor_obs = batch["actor_obs"][idx]
                critic_obs = batch["critic_obs"][idx]
                teacher_obs = self._select_teacher_obs(actor_obs, critic_obs)
                actions = batch["actions"][idx]
                old_log_probs = batch["log_probs"][idx]
                adv = batch["advantages"][idx]
                returns = batch["returns"][idx]

                log_probs, entropy = self.teacher_policy.evaluate_actions(teacher_obs, actions)
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
        target_action_abs = []
        student_action_abs = []

        for _ in range(self.cfg.student_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                actor_obs = batch["actor_obs"][idx]
                teacher_targets = batch["actions"][idx]

                student_actions = self.student_policy.deterministic(actor_obs)
                bc_loss = self.cfg.student_bc_weight * (student_actions - teacher_targets).square().mean()

                self.student_optimizer.zero_grad(set_to_none=True)
                bc_loss.backward()
                nn.utils.clip_grad_norm_(self.student_policy.parameters(), self.cfg.max_grad_norm)
                self.student_optimizer.step()

                losses.append(bc_loss.detach())
                target_action_abs.append(teacher_targets.abs().mean().detach())
                student_action_abs.append(student_actions.abs().mean().detach())

        return {
            "student_bc_loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
            "teacher_target_action_abs_mean": float(torch.stack(target_action_abs).mean().item()) if target_action_abs else 0.0,
            "student_action_abs_mean": float(torch.stack(student_action_abs).mean().item()) if student_action_abs else 0.0,
        }

    def deployment_actions(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.student_policy.deterministic(actor_obs)

    def teacher_actions(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor | None = None,
        sample: bool = False,
    ) -> torch.Tensor:
        teacher_obs = self._select_teacher_obs(actor_obs, critic_obs)
        if sample:
            actions, _ = self.teacher_policy.sample(teacher_obs)
            return actions
        return self.teacher_policy.deterministic(teacher_obs)

    def _checkpoint_payload(self, stage: str) -> dict[str, Any]:
        payload = {
            "stage": stage,
            "teacher_iteration": self.teacher_iteration,
            "student_iteration": self.student_iteration,
            "config": asdict(self.cfg),
            "teacher_policy": self.teacher_policy.state_dict(),
            "student_policy": self.student_policy.state_dict(),
            "value": self.value.state_dict(),
            "teacher_optimizer": self.teacher_optimizer.state_dict(),
            "student_optimizer": self.student_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        return self._detach_to_cpu(payload)

    @classmethod
    def _detach_to_cpu(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", copy=True)
        if isinstance(value, dict):
            return {key: cls._detach_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._detach_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._detach_to_cpu(item) for item in value)
        return value

    def _write_checkpoint(self, payload: dict[str, Any], tmp_path: Path, checkpoint_path: Path) -> None:
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}", flush=True)
        except BaseException as exc:
            self._checkpoint_error = exc
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"Checkpoint save failed for {checkpoint_path}: {exc}", flush=True)

    def _clear_finished_checkpoint_thread(self) -> None:
        if self._checkpoint_thread is not None and not self._checkpoint_thread.is_alive():
            self._checkpoint_thread.join(timeout=0.0)
            self._checkpoint_thread = None
        if self._checkpoint_error is not None:
            error = self._checkpoint_error
            self._checkpoint_error = None
            raise RuntimeError("Previous checkpoint save failed.") from error

    def wait_for_pending_checkpoint(self) -> None:
        if self._checkpoint_thread is None:
            return
        self._checkpoint_thread.join(timeout=self._checkpoint_join_timeout_s)
        if self._checkpoint_thread.is_alive():
            print(
                f"Checkpoint save is still running after {self._checkpoint_join_timeout_s:.0f}s; continuing shutdown without blocking indefinitely.",
                flush=True,
            )
            return
        self._checkpoint_thread = None
        self._clear_finished_checkpoint_thread()

    def save(self, checkpoint_path: str | Path, stage: str) -> None:
        self._clear_finished_checkpoint_thread()
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            print(f"Skipping checkpoint {checkpoint_path}; previous checkpoint save is still running.", flush=True)
            return

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        payload = self._checkpoint_payload(stage=stage)
        self._checkpoint_thread = threading.Thread(
            target=self._write_checkpoint,
            args=(payload, tmp_path, checkpoint_path),
            daemon=True,
        )
        self._checkpoint_thread.start()
        print(f"Checkpoint save started: {checkpoint_path}", flush=True)

    def load(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.teacher_iteration = int(payload.get("teacher_iteration", 0))
        self.student_iteration = int(payload.get("student_iteration", 0))
        self.teacher_policy.load_state_dict(payload["teacher_policy"])
        self.student_policy.load_state_dict(payload["student_policy"])
        self.value.load_state_dict(payload["value"])
        if "teacher_optimizer" in payload:
            self.teacher_optimizer.load_state_dict(payload["teacher_optimizer"])
        if "student_optimizer" in payload:
            self.student_optimizer.load_state_dict(payload["student_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])

    def load_teacher(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.teacher_iteration = int(payload.get("teacher_iteration", 0))
        self.teacher_policy.load_state_dict(payload["teacher_policy"])
        self.value.load_state_dict(payload["value"])
        if "teacher_optimizer" in payload:
            self.teacher_optimizer.load_state_dict(payload["teacher_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
