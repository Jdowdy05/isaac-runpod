from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal


def _atanh(value: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    clipped = torch.clamp(value, min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))


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
        squash_actions: bool = False,
    ) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.mean_net = build_mlp(obs_dim, hidden_dims, act_dim, act, output_scale=output_init_scale)
        self.register_buffer("exploration_std", torch.full((act_dim,), exploration_std))
        self.squash_actions = bool(squash_actions)

    def set_exploration_std(self, std: float) -> None:
        self.exploration_std.fill_(float(std))

    def _mean_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net(obs)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self._mean_logits(obs)
        return torch.tanh(mean) if self.squash_actions else mean

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean_logits = self._mean_logits(obs)
        std = self.exploration_std.unsqueeze(0).expand_as(mean_logits)
        return Normal(mean_logits, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        if not self.squash_actions:
            action = dist.rsample()
            return action, dist.log_prob(action).sum(dim=-1)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
        log_prob = log_prob - torch.log(torch.clamp(1.0 - action.square(), min=1.0e-6)).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        if not self.squash_actions:
            return dist.log_prob(actions).sum(dim=-1), dist.entropy().sum(dim=-1)
        clipped_actions = torch.clamp(actions, min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
        pre_tanh = _atanh(clipped_actions)
        log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
        log_prob = log_prob - torch.log(torch.clamp(1.0 - clipped_actions.square(), min=1.0e-6)).sum(dim=-1)
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
        squash_actions: bool = False,
    ) -> None:
        super().__init__()
        if obs_dim % history_steps != 0:
            raise ValueError(
                f"TemporalStudentPolicy expected obs_dim divisible by history_steps, got {obs_dim} and {history_steps}."
            )

        act = resolve_activation(activation)
        self.history_steps = history_steps
        self.frame_dim = obs_dim // history_steps
        self.squash_actions = bool(squash_actions)
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
        action = self.head(features)
        return torch.tanh(action) if self.squash_actions else action

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
