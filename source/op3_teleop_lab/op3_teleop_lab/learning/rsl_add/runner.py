from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner

from op3_teleop_lab.learning.add.config import ADDTrainingConfig
from op3_teleop_lab.learning.rsl_add.algorithm import RslAddPPO


class RslAddOnPolicyRunner(OnPolicyRunner):
    """Isaac Lab 2.3.2 compatible RSL-RL runner with online ADD updates."""

    def __init__(
        self,
        env,
        train_cfg: dict[str, Any],
        *,
        add_cfg: ADDTrainingConfig,
        diff_dim: int,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        runner_cfg = deepcopy(train_cfg)
        runner_cfg.setdefault("algorithm", {})
        runner_cfg["algorithm"]["add_cfg"] = asdict(add_cfg)
        runner_cfg["algorithm"]["diff_dim"] = int(diff_dim)
        super().__init__(env=env, train_cfg=runner_cfg, log_dir=log_dir, device=device)

    def _construct_algorithm(self, obs) -> RslAddPPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` in the `policy` config instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        add_cfg_raw = self.alg_cfg.pop("add_cfg", None)
        if add_cfg_raw is None:
            raise ValueError("RslAddOnPolicyRunner requires algorithm.add_cfg in the runner configuration.")
        add_cfg = add_cfg_raw if isinstance(add_cfg_raw, ADDTrainingConfig) else ADDTrainingConfig.from_dict(add_cfg_raw)
        diff_dim = int(self.alg_cfg.pop("diff_dim"))

        alg = RslAddPPO(
            actor_critic,
            add_cfg=add_cfg,
            diff_dim=diff_dim,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        saved_dict.update(self.alg.get_extra_state_dict())
        torch.save(saved_dict, path)

        if getattr(self, "logger_type", None) in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        self.alg.load_extra_state_dict(loaded_dict)

        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def train_mode(self):
        super().train_mode()
        self.alg.discriminator.train()

    def eval_mode(self):
        super().eval_mode()
        self.alg.discriminator.eval()
