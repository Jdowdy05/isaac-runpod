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
    actor_hidden_dims: tuple[int, ...] = (1024, 512)
    critic_hidden_dims: tuple[int, ...] = (1024, 512)
    disc_hidden_dims: tuple[int, ...] = (1024, 512)
    activation: str = "relu"
    fixed_action_std: float = 0.05

    rollout_steps: int = 32
    max_iterations: int = 5000
    actor_epochs: int = 5
    critic_epochs: int = 2
    disc_epochs: int = 2
    minibatch_size: int = 16384

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

    task_reward_weight: float = 0.0
    disc_reward_weight: float = 1.0

    actor_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
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
        for field_name in ("actor_hidden_dims", "critic_hidden_dims", "disc_hidden_dims"):
            if field_name in data:
                setattr(config, field_name, tuple(data[field_name]))

        for field_name in (
            "activation",
            "fixed_action_std",
            "rollout_steps",
            "max_iterations",
            "actor_epochs",
            "critic_epochs",
            "disc_epochs",
            "minibatch_size",
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
            "save_interval",
            "log_interval",
            "seed",
        ):
            if field_name in data:
                setattr(config, field_name, data[field_name])

        for opt_name in ("actor_optimizer", "critic_optimizer", "disc_optimizer"):
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

