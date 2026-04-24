from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from rsl_rl.runners import OnPolicyRunner

from op3_teleop_lab.learning.add.config import ADDTrainingConfig


class RslAddOnPolicyRunner(OnPolicyRunner):
    """Stock RSL-RL runner with the custom RSL-ADD algorithm wired in."""

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
        runner_cfg["algorithm"]["class_name"] = "op3_teleop_lab.learning.rsl_add.algorithm:RslAddPPO"
        runner_cfg["algorithm"]["add_cfg"] = asdict(add_cfg)
        runner_cfg["algorithm"]["diff_dim"] = int(diff_dim)
        super().__init__(env=env, train_cfg=runner_cfg, log_dir=log_dir, device=device)
