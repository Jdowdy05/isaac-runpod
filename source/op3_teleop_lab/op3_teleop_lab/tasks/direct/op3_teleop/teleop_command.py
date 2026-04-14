from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import SEGMENT_INDEX, TRACKED_SEGMENTS


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-6)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    quat = torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
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
    target_lin_vel_xy: torch.Tensor

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
    - optional `target_lin_vel_xy`: [T, 2]
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

    def _load_dataset(self, dataset_path: str) -> dict[str, torch.Tensor]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")

        data = np.load(path)
        positions = torch.as_tensor(data["positions"], dtype=torch.float32, device=self.device)
        orientations = torch.as_tensor(data["orientations"], dtype=torch.float32, device=self.device)
        position_valid = torch.as_tensor(data["position_valid"], dtype=torch.bool, device=self.device)
        rotation_valid = torch.as_tensor(
            data["rotation_valid"] if "rotation_valid" in data else np.zeros_like(data["position_valid"]),
            dtype=torch.bool,
            device=self.device,
        )
        target_lin_vel_xy = torch.as_tensor(
            data["target_lin_vel_xy"] if "target_lin_vel_xy" in data else np.zeros((len(positions), 2)),
            dtype=torch.float32,
            device=self.device,
        )
        sequence_starts = (
            torch.as_tensor(data["sequence_starts"], dtype=torch.long, device=self.device)
            if "sequence_starts" in data
            else None
        )
        sequence_lengths = (
            torch.as_tensor(data["sequence_lengths"], dtype=torch.long, device=self.device)
            if "sequence_lengths" in data
            else None
        )

        return {
            "positions": positions,
            "orientations": _normalize_quat(orientations),
            "position_valid": position_valid,
            "rotation_valid": rotation_valid,
            "target_lin_vel_xy": target_lin_vel_xy,
            "sequence_starts": sequence_starts,
            "sequence_lengths": sequence_lengths,
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.phase[env_ids] = torch.rand(len(env_ids), device=self.device) * (2.0 * torch.pi)
        if self.dataset is not None:
            if self.dataset["sequence_starts"] is not None:
                self.sequence_ids[env_ids] = torch.randint(0, self.num_sequences, (len(env_ids),), device=self.device)
                self.sequence_offsets[env_ids] = 0
            else:
                self.frame_idx[env_ids] = torch.randint(0, self.max_frame, (len(env_ids),), device=self.device)

    def step(self) -> SparsePoseBatch:
        if self.dataset is not None:
            batch = self._dataset_batch()
        else:
            batch = self._synthetic_batch()
        self.phase = torch.remainder(self.phase + self.dt * 2.5, 2.0 * torch.pi)
        return batch

    def _dataset_batch(self) -> SparsePoseBatch:
        if self.dataset["sequence_starts"] is not None:
            seq_starts = self.dataset["sequence_starts"][self.sequence_ids]
            seq_lengths = self.dataset["sequence_lengths"][self.sequence_ids]
            idx = seq_starts + self.sequence_offsets
        else:
            idx = self.frame_idx
        positions = self.dataset["positions"][idx]
        orientations = self.dataset["orientations"][idx]
        position_valid = self.dataset["position_valid"][idx]
        rotation_valid = self.dataset["rotation_valid"][idx]
        target_lin_vel_xy = self.dataset["target_lin_vel_xy"][idx]
        if self.dataset["sequence_starts"] is not None:
            self.sequence_offsets += 1
            done = self.sequence_offsets >= seq_lengths
            if torch.any(done):
                done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                self.sequence_ids[done_ids] = torch.randint(0, self.num_sequences, (len(done_ids),), device=self.device)
                self.sequence_offsets[done_ids] = 0
        else:
            self.frame_idx = torch.remainder(self.frame_idx + 1, self.max_frame)
        return SparsePoseBatch(
            positions,
            orientations,
            position_valid,
            rotation_valid,
            self.phase.clone(),
            target_lin_vel_xy,
        )

    def _synthetic_batch(self) -> SparsePoseBatch:
        batch = self.num_envs
        positions = torch.zeros(batch, self.num_segments, 3, dtype=torch.float32, device=self.device)
        orientations = torch.zeros(batch, self.num_segments, 4, dtype=torch.float32, device=self.device)
        position_valid = torch.ones(batch, self.num_segments, dtype=torch.bool, device=self.device)
        rotation_valid = torch.ones(batch, self.num_segments, dtype=torch.bool, device=self.device)
        target_lin_vel_xy = torch.zeros(batch, 2, dtype=torch.float32, device=self.device)

        phase = self.phase
        swing = torch.sin(phase)
        anti_swing = torch.sin(phase + torch.pi)
        torso_yaw = 0.15 * torch.sin(phase * 0.5)

        positions[:, SEGMENT_INDEX["pelvis"]] = torch.tensor((0.0, 0.0, 0.0), device=self.device)
        positions[:, SEGMENT_INDEX["head"]] = torch.stack(
            (0.02 * torch.sin(phase * 0.3), torch.zeros_like(phase), 0.30 + 0.01 * torch.sin(phase)), dim=-1
        )
        positions[:, SEGMENT_INDEX["left_hand"]] = torch.stack(
            (0.18 + 0.05 * anti_swing, 0.16, 0.14 + 0.03 * torch.cos(phase)), dim=-1
        )
        positions[:, SEGMENT_INDEX["right_hand"]] = torch.stack(
            (0.18 + 0.05 * swing, -0.16, 0.14 + 0.03 * torch.cos(phase + torch.pi)), dim=-1
        )
        positions[:, SEGMENT_INDEX["left_knee"]] = torch.stack((0.02 + 0.05 * swing, 0.05, -0.22), dim=-1)
        positions[:, SEGMENT_INDEX["right_knee"]] = torch.stack((0.02 + 0.05 * anti_swing, -0.05, -0.22), dim=-1)
        positions[:, SEGMENT_INDEX["left_foot"]] = torch.stack(
            (0.05 + 0.10 * swing, 0.05, -0.44 + 0.05 * torch.clamp(swing, min=0.0)), dim=-1
        )
        positions[:, SEGMENT_INDEX["right_foot"]] = torch.stack(
            (0.05 + 0.10 * anti_swing, -0.05, -0.44 + 0.05 * torch.clamp(anti_swing, min=0.0)), dim=-1
        )

        identity = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).repeat(batch, 1)
        orientations[:] = identity.unsqueeze(1)
        orientations[:, SEGMENT_INDEX["pelvis"]] = quat_from_euler_xyz(
            torch.zeros_like(phase), 0.02 * torch.sin(phase), torso_yaw
        )
        orientations[:, SEGMENT_INDEX["head"]] = quat_from_euler_xyz(
            torch.zeros_like(phase), 0.08 * torch.sin(phase * 0.5), 0.05 * torch.sin(phase * 0.25)
        )
        orientations[:, SEGMENT_INDEX["left_hand"]] = quat_from_euler_xyz(
            0.25 * anti_swing, torch.zeros_like(phase), 0.15 * anti_swing
        )
        orientations[:, SEGMENT_INDEX["right_hand"]] = quat_from_euler_xyz(
            0.25 * swing, torch.zeros_like(phase), -0.15 * swing
        )

        target_lin_vel_xy[:, 0] = 0.25 + 0.10 * torch.sin(phase * 0.25)
        return SparsePoseBatch(
            positions,
            orientations,
            position_valid,
            rotation_valid,
            phase.clone(),
            target_lin_vel_xy,
        )
