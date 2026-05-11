from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimizerConfig:
    type: str = "adam"
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0


@dataclass
class TeacherStudentTrainingConfig:
    teacher_hidden_dims: tuple[int, ...] = (512, 256, 128)
    student_hidden_dims: tuple[int, ...] = (512, 256)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    teacher_output_init_scale: float = 0.1
    student_output_init_scale: float = 0.1
    teacher_exploration_std: float = 0.05
    teacher_exploration_final_std: float = 0.01
    teacher_exploration_decay_iterations: int = 50000
    teacher_uses_critic_obs: bool = True
    squash_actions: bool = True
    student_rnn_hidden_dim: int = 256

    rollout_steps: int = 24
    teacher_max_iterations: int = 200000
    student_max_iterations: int = 50000
    teacher_epochs: int = 5
    critic_epochs: int = 5
    student_epochs: int = 5
    minibatch_size: int = 16384
    student_batch_size: int = 16384

    discount: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.2
    student_bc_weight: float = 1.0

    teacher_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    student_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=3.0e-4)
    )
    critic_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    save_interval: int = 500
    log_interval: int = 10
    seed: int = 1

    @staticmethod
    def from_yaml(path: str | Path) -> "TeacherStudentTrainingConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TeacherStudentTrainingConfig.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TeacherStudentTrainingConfig":
        config = TeacherStudentTrainingConfig()

        for field_name in ("teacher_hidden_dims", "student_hidden_dims", "critic_hidden_dims"):
            if field_name in data:
                setattr(config, field_name, tuple(data[field_name]))

        for field_name in (
            "activation",
            "teacher_output_init_scale",
            "student_output_init_scale",
            "teacher_exploration_std",
            "teacher_exploration_final_std",
            "teacher_exploration_decay_iterations",
            "teacher_uses_critic_obs",
            "squash_actions",
            "student_rnn_hidden_dim",
            "rollout_steps",
            "teacher_max_iterations",
            "student_max_iterations",
            "teacher_epochs",
            "critic_epochs",
            "student_epochs",
            "minibatch_size",
            "student_batch_size",
            "discount",
            "gae_lambda",
            "ppo_clip_ratio",
            "value_loss_coef",
            "entropy_coef",
            "max_grad_norm",
            "student_bc_weight",
            "save_interval",
            "log_interval",
            "seed",
        ):
            if field_name in data:
                setattr(config, field_name, data[field_name])

        for opt_name in ("teacher_optimizer", "student_optimizer", "critic_optimizer"):
            if opt_name in data:
                opt_data = data[opt_name]
                setattr(
                    config,
                    opt_name,
                    OptimizerConfig(
                        type=opt_data.get("type", "adam"),
                        learning_rate=opt_data.get("learning_rate", 1.0e-3),
                        weight_decay=opt_data.get("weight_decay", 0.0),
                    ),
                )

        return config
