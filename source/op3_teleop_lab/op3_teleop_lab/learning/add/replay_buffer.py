from __future__ import annotations

import torch


class TensorReplayBuffer:
    def __init__(self, capacity: int, feature_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.device = device
        self.storage = torch.zeros((capacity, feature_dim), device=device, dtype=torch.float32)
        self.size = 0
        self.ptr = 0

    def add(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, self.feature_dim).detach()
        count = flat.shape[0]
        if count >= self.capacity:
            self.storage[:] = flat[-self.capacity :]
            self.size = self.capacity
            self.ptr = 0
            return

        end = self.ptr + count
        if end <= self.capacity:
            self.storage[self.ptr : end] = flat
        else:
            first = self.capacity - self.ptr
            self.storage[self.ptr :] = flat[:first]
            self.storage[: end - self.capacity] = flat[first:]
        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, count: int) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = torch.randint(0, self.size, (count,), device=self.device)
        return self.storage[idx]

