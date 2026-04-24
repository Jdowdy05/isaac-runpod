from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import SEGMENT_INDEX, TRACKED_SEGMENTS


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-6)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., :3] *= -1.0
    return result


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * t + torch.cross(quat_xyz, t, dim=-1)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    quat = torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )
    return _normalize_quat(quat)


@dataclass
class SparsePoseBatch:
    positions: torch.Tensor
    orientations: torch.Tensor
    position_valid: torch.Tensor
    rotation_valid: torch.Tensor
    phase: torch.Tensor

    def flatten(self) -> torch.Tensor:
        poses = torch.cat((self.positions, self.orientations), dim=-1).reshape(self.positions.shape[0], -1)
        position_valid = self.position_valid.to(dtype=poses.dtype).reshape(self.positions.shape[0], -1)
        rotation_valid = self.rotation_valid.to(dtype=poses.dtype).reshape(self.positions.shape[0], -1)
        return torch.cat((poses, position_valid, rotation_valid), dim=-1)


class SparsePoseCommandGenerator:
    """Produces sparse human pose commands for training.

    `synthetic` is intended for bring-up and smoke testing.
    `dataset` expects an NPZ file containing:

    - `positions`: [T, num_segments, 3]
    - `orientations`: [T, num_segments, 4]
    - `position_valid`: [T, num_segments]
    - `rotation_valid`: [T, num_segments]
    Dataset positions are accepted as pelvis-origin deltas and normalized to the
    pelvis coordinate frame at load time. Non-pelvis orientations are expected
    to be pelvis-relative; pelvis orientation is treated as unavailable for the
    deployable command.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dt: float,
        mode: str = "synthetic",
        dataset_path: str | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mode = mode
        self.num_segments = len(TRACKED_SEGMENTS)

        self.phase = torch.rand(self.num_envs, device=self.device) * (2.0 * torch.pi)
        self.frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sequence_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sequence_offsets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.dataset = None
        if self.mode == "dataset":
            if not dataset_path:
                raise ValueError("dataset mode requires teleop_dataset_path")
            self.dataset = self._load_dataset(dataset_path)
            self.max_frame = int(self.dataset["positions"].shape[0])
            if self.dataset["sequence_starts"] is not None:
                self.num_sequences = int(self.dataset["sequence_starts"].shape[0])
                self.sequence_ids = torch.randint(0, self.num_sequences, (self.num_envs,), device=self.device)
                self.sequence_offsets.zero_()
            else:
                self.frame_idx = torch.randint(0, self.max_frame, (self.num_envs,), device=self.device)

    def _load_dataset(self, dataset_path: str) -> dict[str, torch.Tensor | None]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")

        data = np.load(path)
        raw_positions = torch.as_tensor(data["positions"], dtype=torch.float32, device=self.device)
        raw_orientations = torch.as_tensor(data["orientations"], dtype=torch.float32, device=self.device)
        if raw_positions.ndim != 3 or raw_positions.shape[1:] != (self.num_segments, 3):
            raise ValueError(
                f"Expected positions with shape [T, {self.num_segments}, 3], got {tuple(raw_positions.shape)}"
            )
        if raw_orientations.shape != (raw_positions.shape[0], self.num_segments, 4):
            raise ValueError(
                f"Expected orientations with shape [T, {self.num_segments}, 4], got {tuple(raw_orientations.shape)}"
            )

        position_valid = torch.as_tensor(data["position_valid"], dtype=torch.bool, device=self.device)
        if "rotation_valid" not in data:
            raise ValueError(
                "Dataset mode requires rotation_valid so pelvis-origin positions can be normalized into the pelvis frame."
            )
        rotation_valid = torch.as_tensor(
            data["rotation_valid"],
            dtype=torch.bool,
            device=self.device,
        )
        if position_valid.shape != raw_positions.shape[:2]:
            raise ValueError(
                f"Expected position_valid with shape {tuple(raw_positions.shape[:2])}, got {tuple(position_valid.shape)}"
            )
        if rotation_valid.shape != raw_positions.shape[:2]:
            raise ValueError(
                f"Expected rotation_valid with shape {tuple(raw_positions.shape[:2])}, got {tuple(rotation_valid.shape)}"
            )

        position_valid = position_valid & torch.isfinite(raw_positions).all(dim=-1)
        positions = torch.nan_to_num(raw_positions, nan=0.0, posinf=0.0, neginf=0.0)

        orientation_finite = torch.isfinite(raw_orientations).all(dim=-1)
        orientations = torch.nan_to_num(raw_orientations, nan=0.0, posinf=0.0, neginf=0.0)
        orientation_norm = torch.linalg.norm(orientations, dim=-1, keepdim=True)
        orientation_ok = orientation_finite & (orientation_norm[..., 0] > 1.0e-6)
        rotation_valid = rotation_valid & orientation_ok
        identity = torch.zeros_like(orientations)
        identity[..., 3] = 1.0
        orientations = torch.where(orientation_ok.unsqueeze(-1), _normalize_quat(orientations), identity)

        pelvis_idx = SEGMENT_INDEX["pelvis"]
        pelvis_position_valid = position_valid[:, pelvis_idx]
        pelvis_rotation_valid = rotation_valid[:, pelvis_idx]
        pelvis_quat_inv = quat_conjugate(orientations[:, pelvis_idx]).unsqueeze(1).expand(-1, self.num_segments, -1)
        positions = quat_apply(pelvis_quat_inv, positions)
        positions[:, pelvis_idx] = 0.0
        position_valid = position_valid & pelvis_rotation_valid.unsqueeze(-1)
        position_valid[:, pelvis_idx] = pelvis_position_valid
        orientations[:, pelvis_idx] = identity[:, pelvis_idx]
        rotation_valid[:, pelvis_idx] = False

        has_sequence_starts = "sequence_starts" in data
        has_sequence_lengths = "sequence_lengths" in data
        if has_sequence_starts != has_sequence_lengths:
            raise ValueError("Sparse dataset must contain both sequence_starts and sequence_lengths, or neither.")
        sequence_starts = (
            torch.as_tensor(data["sequence_starts"], dtype=torch.long, device=self.device)
            if has_sequence_starts
            else None
        )
        sequence_lengths = (
            torch.as_tensor(data["sequence_lengths"], dtype=torch.long, device=self.device)
            if has_sequence_lengths
            else None
        )
        if sequence_starts is not None and sequence_lengths is not None:
            if (
                sequence_starts.ndim != 1
                or sequence_lengths.ndim != 1
                or sequence_starts.shape != sequence_lengths.shape
            ):
                raise ValueError("sequence_starts and sequence_lengths must be matching 1-D arrays.")
            if torch.any(sequence_lengths <= 0):
                raise ValueError("sequence_lengths must be strictly positive.")
            if torch.any(sequence_starts < 0) or torch.any(
                sequence_starts + sequence_lengths > raw_positions.shape[0]
            ):
                raise ValueError("sequence_starts/sequence_lengths contain ranges outside the sparse dataset.")

        return {
            "positions": positions,
            "orientations": orientations,
            "position_valid": position_valid,
            "rotation_valid": rotation_valid,
            "sequence_starts": sequence_starts,
            "sequence_lengths": sequence_lengths,
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self.dataset is None:
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device) * (2.0 * torch.pi)
        self.command_done[env_ids] = False
        if self.dataset is not None:
            if self.dataset["sequence_starts"] is not None:
                self.sequence_ids[env_ids] = torch.randint(0, self.num_sequences, (len(env_ids),), device=self.device)
                self.sequence_offsets[env_ids] = 0
            else:
                self.frame_idx[env_ids] = torch.randint(0, self.max_frame, (len(env_ids),), device=self.device)

    def step(self) -> SparsePoseBatch:
        if self.dataset is not None:
            return self._dataset_batch()
        else:
            self.command_done.zero_()
            batch = self._synthetic_batch()
        self.phase = torch.remainder(self.phase + self.dt * 2.5, 2.0 * torch.pi)
        return batch

    def _dataset_batch(self) -> SparsePoseBatch:
        self.command_done.zero_()
        if self.dataset["sequence_starts"] is not None:
            seq_starts = self.dataset["sequence_starts"][self.sequence_ids]
            seq_lengths = self.dataset["sequence_lengths"][self.sequence_ids]
            safe_offsets = torch.minimum(self.sequence_offsets, seq_lengths - 1)
            idx = seq_starts + safe_offsets
            phase = (2.0 * torch.pi) * safe_offsets.float() / torch.clamp(seq_lengths.float() - 1.0, min=1.0)
        else:
            idx = self.frame_idx
            denom = max(self.max_frame - 1, 1)
            phase = (2.0 * torch.pi) * self.frame_idx.float() / float(denom)
        positions = self.dataset["positions"][idx]
        orientations = self.dataset["orientations"][idx]
        position_valid = self.dataset["position_valid"][idx]
        rotation_valid = self.dataset["rotation_valid"][idx]
        if self.dataset["sequence_starts"] is not None:
            next_offsets = self.sequence_offsets + 1
            self.command_done.copy_(next_offsets >= seq_lengths)
            self.sequence_offsets = torch.minimum(next_offsets, seq_lengths)
        else:
            self.frame_idx = torch.remainder(self.frame_idx + 1, self.max_frame)
        return SparsePoseBatch(
            positions,
            orientations,
            position_valid,
            rotation_valid,
            phase,
        )

    def _synthetic_batch(self) -> SparsePoseBatch:
        batch = self.num_envs
        positions = torch.zeros(batch, self.num_segments, 3, dtype=torch.float32, device=self.device)
        orientations = torch.zeros(batch, self.num_segments, 4, dtype=torch.float32, device=self.device)
        position_valid = torch.ones(batch, self.num_segments, dtype=torch.bool, device=self.device)
        rotation_valid = torch.ones(batch, self.num_segments, dtype=torch.bool, device=self.device)

        phase = self.phase
        swing = torch.sin(phase)
        anti_swing = torch.sin(phase + torch.pi)
        zeros = torch.zeros_like(phase)

        positions[:, SEGMENT_INDEX["pelvis"]] = torch.tensor((0.0, 0.0, 0.0), device=self.device)
        positions[:, SEGMENT_INDEX["head"]] = torch.stack(
            (0.02 * torch.sin(phase * 0.3), zeros, 0.30 + 0.01 * torch.sin(phase)), dim=-1
        )
        positions[:, SEGMENT_INDEX["left_hand"]] = torch.stack(
            (0.18 + 0.05 * anti_swing, torch.full_like(phase, 0.16), 0.14 + 0.03 * torch.cos(phase)), dim=-1
        )
        positions[:, SEGMENT_INDEX["right_hand"]] = torch.stack(
            (0.18 + 0.05 * swing, torch.full_like(phase, -0.16), 0.14 + 0.03 * torch.cos(phase + torch.pi)), dim=-1
        )
        positions[:, SEGMENT_INDEX["left_knee"]] = torch.stack(
            (0.02 + 0.05 * swing, torch.full_like(phase, 0.05), torch.full_like(phase, -0.22)),
            dim=-1,
        )
        positions[:, SEGMENT_INDEX["right_knee"]] = torch.stack(
            (0.02 + 0.05 * anti_swing, torch.full_like(phase, -0.05), torch.full_like(phase, -0.22)),
            dim=-1,
        )
        positions[:, SEGMENT_INDEX["left_foot"]] = torch.stack(
            (
                0.05 + 0.10 * swing,
                torch.full_like(phase, 0.05),
                -0.44 + 0.05 * torch.clamp(swing, min=0.0),
            ),
            dim=-1,
        )
        positions[:, SEGMENT_INDEX["right_foot"]] = torch.stack(
            (
                0.05 + 0.10 * anti_swing,
                torch.full_like(phase, -0.05),
                -0.44 + 0.05 * torch.clamp(anti_swing, min=0.0),
            ),
            dim=-1,
        )

        identity = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device).repeat(batch, 1)
        orientations[:] = identity.unsqueeze(1)
        rotation_valid[:, SEGMENT_INDEX["pelvis"]] = False
        orientations[:, SEGMENT_INDEX["head"]] = quat_from_euler_xyz(
            zeros, 0.08 * torch.sin(phase * 0.5), 0.05 * torch.sin(phase * 0.25)
        )
        orientations[:, SEGMENT_INDEX["left_hand"]] = quat_from_euler_xyz(
            0.25 * anti_swing, zeros, 0.15 * anti_swing
        )
        orientations[:, SEGMENT_INDEX["right_hand"]] = quat_from_euler_xyz(
            0.25 * swing, zeros, -0.15 * swing
        )

        return SparsePoseBatch(
            positions,
            orientations,
            position_valid,
            rotation_valid,
            phase.clone(),
        )
