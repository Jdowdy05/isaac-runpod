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
    ) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.mean_net = build_mlp(obs_dim, hidden_dims, act_dim, act, output_scale=output_init_scale)
        self.register_buffer("exploration_std", torch.full((act_dim,), exploration_std))

    def set_exploration_std(self, std: float) -> None:
        self.exploration_std.fill_(float(std))

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net(obs)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.deterministic(obs)
        std = self.exploration_std.unsqueeze(0).expand_as(mean)
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
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
    ) -> None:
        super().__init__()
        if obs_dim % history_steps != 0:
            raise ValueError(
                f"TemporalStudentPolicy expected obs_dim divisible by history_steps, got {obs_dim} and {history_steps}."
            )

        act = resolve_activation(activation)
        self.history_steps = history_steps
        self.frame_dim = obs_dim // history_steps
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
        return self.head(features)

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
