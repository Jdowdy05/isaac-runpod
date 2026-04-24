from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from op3_teleop_lab.learning.add.config import ADDTrainingConfig
from op3_teleop_lab.learning.add.networks import DifferentialDiscriminator
from op3_teleop_lab.learning.add.normalizers import DiffNormalizer
from op3_teleop_lab.learning.add.replay_buffer import TensorReplayBuffer


class RslAddPPO(PPO):
    """RSL-RL PPO with an online ADD discriminator reward model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
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
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.add_cfg = add_cfg
        self.diff_dim = int(diff_dim)
        self.discriminator = DifferentialDiscriminator(
            diff_dim=self.diff_dim,
            hidden_dims=add_cfg.disc_hidden_dims,
            activation=add_cfg.activation,
        ).to(self.device)
        self.disc_optimizer = self._make_disc_optimizer(add_cfg, self.discriminator.parameters())
        self.diff_normalizer = DiffNormalizer(self.diff_dim, device=torch.device(self.device))
        self.replay_buffer = TensorReplayBuffer(
            add_cfg.disc_replay_capacity,
            self.diff_dim,
            device=torch.device(self.device),
        )
        self.diff_storage = torch.zeros(
            storage.num_transitions_per_env,
            storage.num_envs,
            self.diff_dim,
            device=self.device,
            dtype=torch.float32,
        )

        self.last_task_rewards: torch.Tensor | None = None
        self.last_disc_rewards: torch.Tensor | None = None
        self.last_total_rewards: torch.Tensor | None = None
        self._rollout_task_reward_sum = 0.0
        self._rollout_disc_reward_sum = 0.0
        self._rollout_total_reward_sum = 0.0
        self._rollout_reward_count = 0

    @staticmethod
    def _make_disc_optimizer(add_cfg: ADDTrainingConfig, params) -> torch.optim.Optimizer:
        opt_cfg = add_cfg.disc_optimizer
        opt_type = opt_cfg.type.lower()
        if opt_type == "sgd":
            return torch.optim.SGD(params, lr=opt_cfg.learning_rate, weight_decay=opt_cfg.weight_decay)
        if opt_type == "adam":
            return torch.optim.Adam(params, lr=opt_cfg.learning_rate, weight_decay=opt_cfg.weight_decay)
        raise ValueError(f"Unsupported discriminator optimizer type: {opt_cfg.type}")

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if "add_diff" not in extras:
            raise KeyError("RslAddPPO requires env extras['add_diff'] for ADD discriminator rewards.")

        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        diffs = extras["add_diff"].to(self.device)
        if diffs.shape[-1] != self.diff_dim:
            raise ValueError(f"Expected add_diff dim {self.diff_dim}, got {diffs.shape[-1]}.")
        diffs = torch.nan_to_num(diffs.reshape(-1, self.diff_dim), nan=0.0, posinf=0.0, neginf=0.0)
        self.diff_storage[self.storage.step].copy_(diffs.reshape(self.diff_storage[self.storage.step].shape))

        task_rewards = torch.nan_to_num(rewards.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        disc_rewards = self.compute_disc_rewards(diffs).reshape(-1)
        action_l2_penalty = torch.sum(self.transition.actions.square(), dim=-1)  # type: ignore[arg-type]
        total_rewards = (
            self.add_cfg.task_reward_weight * task_rewards
            + self.add_cfg.disc_reward_weight * disc_rewards
            - self.add_cfg.action_l2_reward_weight * action_l2_penalty
        )
        total_rewards = torch.nan_to_num(total_rewards, nan=0.0, posinf=0.0, neginf=0.0)

        self.transition.rewards = total_rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs.unsqueeze(1),  # type: ignore[operator]
                1,
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

        self.last_task_rewards = task_rewards.detach()
        self.last_disc_rewards = disc_rewards.detach()
        self.last_total_rewards = total_rewards.detach()
        self._rollout_task_reward_sum += float(task_rewards.mean().item())
        self._rollout_disc_reward_sum += float(disc_rewards.mean().item())
        self._rollout_total_reward_sum += float(total_rewards.mean().item())
        self._rollout_reward_count += 1

    def update(self) -> dict[str, float]:
        flat_diffs = self.diff_storage[: self.storage.step].reshape(-1, self.diff_dim)
        flat_diffs = torch.nan_to_num(flat_diffs, nan=0.0, posinf=0.0, neginf=0.0)

        loss_dict = super().update()

        self.diff_normalizer.record(flat_diffs)
        self.diff_normalizer.update()
        disc_stats = self.update_discriminator(flat_diffs)
        loss_dict.update(disc_stats)

        denom = max(self._rollout_reward_count, 1)
        loss_dict["task_reward_mean"] = self._rollout_task_reward_sum / denom
        loss_dict["disc_reward_mean"] = self._rollout_disc_reward_sum / denom
        loss_dict["total_reward_mean"] = self._rollout_total_reward_sum / denom
        self._rollout_task_reward_sum = 0.0
        self._rollout_disc_reward_sum = 0.0
        self._rollout_total_reward_sum = 0.0
        self._rollout_reward_count = 0
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
        if num_samples == 0:
            return {
                "disc_loss": 0.0,
                "disc_grad_penalty": 0.0,
                "disc_pos_acc": 0.0,
                "disc_neg_acc": 0.0,
            }

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

            pos_grad = torch.autograd.grad(
                outputs=pos_logits.sum(),
                inputs=pos_diff,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            neg_grad = torch.autograd.grad(
                outputs=neg_logits.sum(),
                inputs=norm_neg_diff,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_penalty = 0.5 * (
                pos_grad.square().sum(dim=-1).mean() + neg_grad.square().sum(dim=-1).mean()
            )
            logit_reg = self.add_cfg.disc_logit_reg * (
                pos_logits.square().mean() + neg_logits.square().mean()
            )
            total_loss = disc_loss + self.add_cfg.disc_grad_penalty * grad_penalty + logit_reg

            self.disc_optimizer.zero_grad()
            total_loss.backward()
            self.disc_optimizer.step()

            losses.append(float(disc_loss.item()))
            grad_penalties.append(float(grad_penalty.item()))
            pos_accs.append(float((torch.sigmoid(pos_logits) > 0.5).float().mean().item()))
            neg_accs.append(float((torch.sigmoid(neg_logits) < 0.5).float().mean().item()))

        return {
            "disc_loss": sum(losses) / len(losses),
            "disc_grad_penalty": sum(grad_penalties) / len(grad_penalties),
            "disc_pos_acc": sum(pos_accs) / len(pos_accs),
            "disc_neg_acc": sum(neg_accs) / len(neg_accs),
        }

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict.update(
            {
                "disc_state_dict": self.discriminator.state_dict(),
                "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
                "diff_normalizer_state_dict": self.diff_normalizer.state_dict(),
                "add_cfg_dict": asdict(self.add_cfg),
                "diff_dim": self.diff_dim,
            }
        )
        if self.replay_buffer.size > 0:
            saved_dict["disc_replay_buffer"] = {
                "storage": self.replay_buffer.storage[: self.replay_buffer.size].detach().cpu(),
                "size": self.replay_buffer.size,
                "ptr": self.replay_buffer.ptr,
            }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "disc": True,
            }

        if load_cfg.get("disc"):
            if "disc_state_dict" in loaded_dict:
                self.discriminator.load_state_dict(loaded_dict["disc_state_dict"], strict=strict)
            if "disc_optimizer_state_dict" in loaded_dict:
                self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
            if "diff_normalizer_state_dict" in loaded_dict:
                self.diff_normalizer.load_state_dict(loaded_dict["diff_normalizer_state_dict"], strict=False)
            replay_state = loaded_dict.get("disc_replay_buffer")
            if replay_state is not None:
                replay_storage = replay_state["storage"].to(self.device)
                count = min(replay_storage.shape[0], self.replay_buffer.capacity)
                self.replay_buffer.storage[:count] = replay_storage[-count:]
                self.replay_buffer.size = count
                self.replay_buffer.ptr = int(replay_state.get("ptr", count % self.replay_buffer.capacity))
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "RslAddPPO":
        alg_class: type[RslAddPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        add_cfg_raw = cfg["algorithm"].pop("add_cfg", None)
        if add_cfg_raw is None:
            raise ValueError("RslAddPPO requires algorithm.add_cfg in the runner configuration.")
        if isinstance(add_cfg_raw, ADDTrainingConfig):
            add_cfg = add_cfg_raw
        else:
            add_cfg = ADDTrainingConfig.from_dict(add_cfg_raw)
        diff_dim = int(cfg["algorithm"].pop("diff_dim"))

        alg = alg_class(
            actor,
            critic,
            storage,
            add_cfg=add_cfg,
            diff_dim=diff_dim,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        return alg
