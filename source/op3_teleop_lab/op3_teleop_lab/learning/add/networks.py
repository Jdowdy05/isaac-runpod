from __future__ import annotations

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


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        fixed_action_std: float,
    ) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.mean_net = build_mlp(obs_dim, hidden_dims, act_dim, act, output_scale=0.01)
        self.register_buffer("std", torch.full((act_dim,), fixed_action_std))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = self.std.unsqueeze(0).expand_as(mean)
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


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

