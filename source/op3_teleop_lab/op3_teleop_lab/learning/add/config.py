from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimizerConfig:
    type: str = "sgd"
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0


@dataclass
class ADDTrainingConfig:
    teacher_hidden_dims: tuple[int, ...] = (1024, 512)
    student_hidden_dims: tuple[int, ...] = (512,)
    critic_hidden_dims: tuple[int, ...] = (1024, 512)
    disc_hidden_dims: tuple[int, ...] = (1024, 512)
    activation: str = "relu"
    teacher_output_init_scale: float = 0.1
    student_output_init_scale: float = 0.1
    action_l2_reward_weight: float = 0.02
    teacher_exploration_std: float = 0.3
    teacher_exploration_final_std: float = 0.08
    teacher_exploration_decay_iterations: int = 30000
    teacher_uses_critic_obs: bool = False
    student_rnn_hidden_dim: int = 256

    rollout_steps: int = 32
    max_iterations: int = 5000
    teacher_epochs: int = 5
    critic_epochs: int = 2
    disc_epochs: int = 2
    student_epochs: int = 2
    minibatch_size: int = 16384
    student_batch_size: int = 16384

    discount: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0

    disc_reward_scale: float = 2.0
    disc_grad_penalty: float = 2.0
    disc_logit_reg: float = 0.01
    disc_replay_capacity: int = 200000
    disc_replay_samples: int = 1000

    task_reward_weight: float = 0.05
    disc_reward_weight: float = 1.0
    student_bc_weight: float = 1.0

    teacher_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    student_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    critic_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    disc_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            type="sgd",
            learning_rate=2.5e-4,
            weight_decay=1.0e-4,
        )
    )

    save_interval: int = 100
    log_interval: int = 10
    seed: int = 42

    @staticmethod
    def from_yaml(path: str | Path) -> "ADDTrainingConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return ADDTrainingConfig.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ADDTrainingConfig":
        config = ADDTrainingConfig()
        if "actor_hidden_dims" in data and "teacher_hidden_dims" not in data:
            data = dict(data)
            data["teacher_hidden_dims"] = data["actor_hidden_dims"]
        if "fixed_action_std" in data and "teacher_exploration_std" not in data:
            data = dict(data)
            data["teacher_exploration_std"] = data["fixed_action_std"]
        if "actor_epochs" in data and "teacher_epochs" not in data:
            data = dict(data)
            data["teacher_epochs"] = data["actor_epochs"]
        if "actor_optimizer" in data and "teacher_optimizer" not in data:
            data = dict(data)
            data["teacher_optimizer"] = data["actor_optimizer"]

        for field_name in ("teacher_hidden_dims", "student_hidden_dims", "critic_hidden_dims", "disc_hidden_dims"):
            if field_name in data:
                setattr(config, field_name, tuple(data[field_name]))

        for field_name in (
            "activation",
            "teacher_output_init_scale",
            "student_output_init_scale",
            "action_l2_reward_weight",
            "teacher_exploration_std",
            "teacher_exploration_final_std",
            "teacher_exploration_decay_iterations",
            "teacher_uses_critic_obs",
            "student_rnn_hidden_dim",
            "rollout_steps",
            "max_iterations",
            "teacher_epochs",
            "critic_epochs",
            "disc_epochs",
            "student_epochs",
            "minibatch_size",
            "student_batch_size",
            "discount",
            "gae_lambda",
            "ppo_clip_ratio",
            "value_loss_coef",
            "entropy_coef",
            "max_grad_norm",
            "disc_reward_scale",
            "disc_grad_penalty",
            "disc_logit_reg",
            "disc_replay_capacity",
            "disc_replay_samples",
            "task_reward_weight",
            "disc_reward_weight",
            "student_bc_weight",
            "save_interval",
            "log_interval",
            "seed",
        ):
            if field_name in data:
                setattr(config, field_name, data[field_name])

        for opt_name in ("teacher_optimizer", "student_optimizer", "critic_optimizer", "disc_optimizer"):
            if opt_name in data:
                opt_data = data[opt_name]
                setattr(
                    config,
                    opt_name,
                    OptimizerConfig(
                        type=opt_data.get("type", "sgd"),
                        learning_rate=opt_data.get("learning_rate", 1.0e-4),
                        weight_decay=opt_data.get("weight_decay", 0.0),
                    ),
                )
        return config
