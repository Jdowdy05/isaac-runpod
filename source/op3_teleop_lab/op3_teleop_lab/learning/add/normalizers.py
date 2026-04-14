from __future__ import annotations

import torch
from torch import nn


class DiffNormalizer(nn.Module):
    def __init__(self, shape: int | tuple[int, ...], device: torch.device, min_diff: float = 1.0e-4) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.min_diff = min_diff
        self.register_buffer("count", torch.zeros(1, device=device, dtype=torch.long))
        self.register_buffer("mean_abs", torch.ones(shape, device=device, dtype=torch.float32))
        self._pending_sum_abs: torch.Tensor | None = None
        self._pending_count = 0

    def record(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, *self.mean_abs.shape)
        batch_sum_abs = torch.sum(torch.abs(flat), dim=0)
        if self._pending_sum_abs is None:
            self._pending_sum_abs = batch_sum_abs
        else:
            self._pending_sum_abs = self._pending_sum_abs + batch_sum_abs
        self._pending_count += flat.shape[0]

    def update(self) -> None:
        if self._pending_sum_abs is None or self._pending_count == 0:
            return
        new_count = torch.tensor([self._pending_count], device=self.mean_abs.device, dtype=torch.long)
        new_mean_abs = self._pending_sum_abs / float(self._pending_count)
        total = self.count + new_count
        old_weight = torch.where(total > 0, self.count.float() / total.float(), torch.zeros_like(total).float())
        new_weight = torch.where(total > 0, new_count.float() / total.float(), torch.zeros_like(total).float())
        self.mean_abs[:] = old_weight * self.mean_abs + new_weight * new_mean_abs
        self.count[:] = total
        self._pending_sum_abs = None
        self._pending_count = 0

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(self.mean_abs, min=self.min_diff)
        return x / denom

