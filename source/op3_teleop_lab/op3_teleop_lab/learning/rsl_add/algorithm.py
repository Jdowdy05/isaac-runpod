from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from op3_teleop_lab.learning.add.config import ADDTrainingConfig
from op3_teleop_lab.learning.add.networks import DifferentialDiscriminator
from op3_teleop_lab.learning.add.normalizers import DiffNormalizer
from op3_teleop_lab.learning.add.replay_buffer import TensorReplayBuffer


class RslAddPPO:
    """RSL-RL PPO with adversarial differential discriminator rewards.

    This keeps the RSL-RL actor-critic/action-noise/storage contract and adds the
    ADD discriminator as an online reward model trained from the rollout's
    differential tracking vectors.
    """

    policy: ActorCritic

    def __init__(
        self,
        policy: ActorCritic,
        add_cfg: ADDTrainingConfig,
        diff_dim: int,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float | None = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        **kwargs,
    ) -> None:
        if kwargs:
            print(f"RslAddPPO ignored unsupported algorithm arguments: {sorted(kwargs)}")

        self.device = device
        self.policy = policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        self.add_cfg = add_cfg
        self.diff_dim = int(diff_dim)
        self.discriminator = DifferentialDiscriminator(
            diff_dim=self.diff_dim,
            hidden_dims=add_cfg.disc_hidden_dims,
            activation=add_cfg.activation,
        ).to(self.device)
        self.disc_optimizer = self._make_disc_optimizer(add_cfg, self.discriminator.parameters())
        self.diff_normalizer = DiffNormalizer(self.diff_dim, device=torch.device(self.device))
        self.replay_buffer = TensorReplayBuffer(add_cfg.disc_replay_capacity, self.diff_dim, device=torch.device(self.device))
        self.diff_storage: torch.Tensor | None = None

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.last_task_rewards: torch.Tensor | None = None
        self.last_disc_rewards: torch.Tensor | None = None
        self.last_total_rewards: torch.Tensor | None = None

    @staticmethod
    def _make_disc_optimizer(add_cfg: ADDTrainingConfig, params) -> torch.optim.Optimizer:
        opt_cfg = add_cfg.disc_optimizer
        opt_type = opt_cfg.type.lower()
        if opt_type == "sgd":
            return torch.optim.SGD(
                params,
                lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
            )
        if opt_type == "adam":
            return torch.optim.Adam(
                params,
                lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
            )
        raise ValueError(f"Unsupported discriminator optimizer type: {opt_cfg.type}")

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape) -> None:
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )
        self.diff_storage = torch.zeros(
            num_transitions_per_env,
            num_envs,
            self.diff_dim,
            device=self.device,
            dtype=torch.float32,
        )

    def act(self, obs):
        if self.storage is None:
            raise RuntimeError("Storage is not initialized.")
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self.storage is None or self.diff_storage is None:
            raise RuntimeError("Storage is not initialized.")
        if "add_diff" not in extras:
            raise KeyError("RslAddPPO requires env extras['add_diff'] for ADD discriminator rewards.")

        self.policy.update_normalization(obs)

        diffs = extras["add_diff"].to(self.device)
        if diffs.shape[-1] != self.diff_dim:
            raise ValueError(f"Expected add_diff dim {self.diff_dim}, got {diffs.shape[-1]}.")
        diffs = torch.nan_to_num(diffs.reshape(-1, self.diff_dim), nan=0.0, posinf=0.0, neginf=0.0)
        step = self.storage.step
        self.diff_storage[step].copy_(diffs.reshape(self.diff_storage[step].shape))

        task_rewards = torch.nan_to_num(rewards.to(self.device).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        dones = dones.to(self.device).reshape(-1)
        disc_rewards = self.compute_disc_rewards(diffs).reshape(-1)
        action_l2_penalty = torch.sum(self.transition.actions.square(), dim=-1)
        total_rewards = (
            self.add_cfg.task_reward_weight * task_rewards
            + self.add_cfg.disc_reward_weight * disc_rewards
            - self.add_cfg.action_l2_reward_weight * action_l2_penalty
        )
        total_rewards = torch.nan_to_num(total_rewards, nan=0.0, posinf=0.0, neginf=0.0)

        self.transition.rewards = total_rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            timeouts = extras["time_outs"].to(self.device).reshape(-1, 1)
            bootstrap = torch.squeeze(self.transition.values * timeouts, 1)
            self.transition.rewards += self.gamma * bootstrap
            self.transition.rewards = torch.nan_to_num(self.transition.rewards, nan=0.0, posinf=0.0, neginf=0.0)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

        self.last_task_rewards = task_rewards.detach()
        self.last_disc_rewards = disc_rewards.detach()
        self.last_total_rewards = total_rewards.detach()

    def compute_returns(self, obs) -> None:
        if self.storage is None:
            raise RuntimeError("Storage is not initialized.")
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self) -> dict[str, float]:
        if self.storage is None or self.diff_storage is None:
            raise RuntimeError("Storage is not initialized.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1.0e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        flat_diffs = self.diff_storage[: self.storage.step].reshape(-1, self.diff_dim)
        flat_diffs = torch.nan_to_num(flat_diffs, nan=0.0, posinf=0.0, neginf=0.0)
        self.diff_normalizer.record(flat_diffs)
        self.diff_normalizer.update()
        disc_stats = self.update_discriminator(flat_diffs)

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            **disc_stats,
        }
        return loss_dict

    def compute_disc_rewards(self, flat_diffs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flat_diffs = torch.nan_to_num(flat_diffs, nan=0.0, posinf=0.0, neginf=0.0)
            norm_diffs = self.diff_normalizer.normalize(flat_diffs)
            norm_diffs = torch.nan_to_num(norm_diffs, nan=0.0, posinf=0.0, neginf=0.0)
            logits = torch.nan_to_num(self.discriminator(norm_diffs), nan=0.0, posinf=0.0, neginf=0.0)
            prob = torch.sigmoid(logits)
            rewards = -torch.log(torch.clamp(1.0 - prob, min=1.0e-4))
            rewards = rewards * self.add_cfg.disc_reward_scale
            return torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

    def update_discriminator(self, flat_diffs: torch.Tensor) -> dict[str, float]:
        flat_diffs = torch.nan_to_num(flat_diffs.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        self.replay_buffer.add(flat_diffs)

        num_samples = flat_diffs.shape[0]
        minibatch_size = min(self.add_cfg.minibatch_size, num_samples)
        steps_per_epoch = max(1, math.ceil(num_samples / minibatch_size))

        losses = []
        grad_penalties = []
        pos_accs = []
        neg_accs = []

        for _ in range(self.add_cfg.disc_epochs * steps_per_epoch):
            idx = torch.randint(0, num_samples, (minibatch_size,), device=self.device)
            current_diff = flat_diffs[idx]
            replay_count = min(current_diff.shape[0], self.add_cfg.disc_replay_samples, self.replay_buffer.size)
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
            disc_loss = disc_loss + self.add_cfg.disc_logit_reg * logit_reg

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
            disc_loss = disc_loss + self.add_cfg.disc_grad_penalty * grad_penalty

            self.disc_optimizer.zero_grad(set_to_none=True)
            disc_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.add_cfg.max_grad_norm)
            self.disc_optimizer.step()

            losses.append(disc_loss.detach())
            grad_penalties.append(grad_penalty.detach())
            pos_accs.append((pos_logits > 0.0).float().mean().detach())
            neg_accs.append((neg_logits < 0.0).float().mean().detach())

        return {
            "disc_loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
            "disc_grad_penalty": float(torch.stack(grad_penalties).mean().item()) if grad_penalties else 0.0,
            "disc_pos_acc": float(torch.stack(pos_accs).mean().item()) if pos_accs else 0.0,
            "disc_neg_acc": float(torch.stack(neg_accs).mean().item()) if neg_accs else 0.0,
        }

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "diff_normalizer": self.diff_normalizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict, load_optimizer: bool = True) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.diff_normalizer.load_state_dict(state_dict["diff_normalizer"])
        if load_optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.disc_optimizer.load_state_dict(state_dict["disc_optimizer"])
