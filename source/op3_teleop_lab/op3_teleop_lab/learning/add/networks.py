from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: type[nn.Module],
    output_scale: float | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    head = nn.Linear(prev_dim, output_dim)
    if output_scale is not None:
        nn.init.uniform_(head.weight, -output_scale, output_scale)
        nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


def resolve_activation(name: str) -> type[nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError(f"Unsupported activation: {name}")


class DeterministicTeacherPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        exploration_std: float,
        output_init_scale: float = 0.1,
        action_bound: float = 1.0,
        sample_action_bound: float | None = 1.0,
    ) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.mean_net = build_mlp(obs_dim, hidden_dims, act_dim, act, output_scale=output_init_scale)
        self.register_buffer("exploration_std", torch.full((act_dim,), exploration_std))
        self.action_bound = float(action_bound)
        self.sample_action_bound = float(sample_action_bound) if sample_action_bound is not None else self.action_bound
        if not math.isclose(self.sample_action_bound, self.action_bound, rel_tol=1.0e-6, abs_tol=1.0e-6):
            raise ValueError(
                "DeterministicTeacherPolicy requires matching action_bound and sample_action_bound when using "
                "the tanh-squashed Gaussian teacher."
            )

    def set_exploration_std(self, std: float) -> None:
        self.exploration_std.fill_(float(std))

    def _mean_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net(obs)

    def _squash(self, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        return self.action_bound * torch.tanh(pre_tanh_action)

    @staticmethod
    def _atanh(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(value) - torch.log1p(-value))

    def _scaled_tanh_log_prob(
        self,
        dist: Normal,
        pre_tanh_action: torch.Tensor,
        squashed_action: torch.Tensor,
    ) -> torch.Tensor:
        safe_squashed = torch.clamp(1.0 - squashed_action.square(), min=1.0e-6)
        log_det_jacobian = math.log(self.action_bound) + torch.log(safe_squashed)
        return (dist.log_prob(pre_tanh_action) - log_det_jacobian).sum(dim=-1)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self._squash(self._mean_logits(obs))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean_logits = self._mean_logits(obs)
        std = self.exploration_std.unsqueeze(0).expand_as(mean_logits)
        return Normal(mean_logits, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        pre_tanh_action = dist.rsample()
        squashed_action = torch.tanh(pre_tanh_action)
        action = self.action_bound * squashed_action
        log_prob = self._scaled_tanh_log_prob(dist, pre_tanh_action, squashed_action)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        squashed_action = torch.clamp(actions / self.action_bound, min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
        pre_tanh_action = self._atanh(squashed_action)
        log_prob = self._scaled_tanh_log_prob(dist, pre_tanh_action, squashed_action)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class TemporalStudentPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        history_steps: int,
        rnn_hidden_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        output_init_scale: float = 0.1,
        action_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if obs_dim % history_steps != 0:
            raise ValueError(
                f"TemporalStudentPolicy expected obs_dim divisible by history_steps, got {obs_dim} and {history_steps}."
            )

        act = resolve_activation(activation)
        self.history_steps = history_steps
        self.frame_dim = obs_dim // history_steps
        self.action_bound = float(action_bound)
        self.gru = nn.GRU(
            input_size=self.frame_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = build_mlp(rnn_hidden_dim, hidden_dims, act_dim, act, output_scale=output_init_scale)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        seq = obs.view(-1, self.history_steps, self.frame_dim)
        _, hidden = self.gru(seq)
        return hidden[-1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self._encode(obs)
        return self.action_bound * torch.tanh(self.head(features))

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Sequence[int], activation: str) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.value_net = build_mlp(obs_dim, hidden_dims, 1, act)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)


class DifferentialDiscriminator(nn.Module):
    def __init__(self, diff_dim: int, hidden_dims: Sequence[int], activation: str) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.backbone = build_mlp(diff_dim, hidden_dims, 1, act, output_scale=1.0)

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        return self.backbone(diff).squeeze(-1)

    def get_logit_weights(self) -> torch.Tensor:
        last_layer = self.backbone[-1]
        assert isinstance(last_layer, nn.Linear)
        return last_layer.weight.view(-1)
