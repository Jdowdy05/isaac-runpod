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


def _read_npz_scalar_string(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Sparse dataset metadata {key!r} must be a scalar string, got shape {value.shape}.")
    return str(value.item()).strip().lower()


@dataclass
class SparsePoseBatch:
    positions: torch.Tensor
    orientations: torch.Tensor
    position_valid: torch.Tensor
    rotation_valid: torch.Tensor
    phase: torch.Tensor
    segment_velocities: torch.Tensor | None = None
    velocity_valid: torch.Tensor | None = None
    root_lin_vel_xy: torch.Tensor | None = None

    def flatten(self) -> torch.Tensor:
        poses = torch.cat((self.positions, self.orientations), dim=-1).reshape(self.positions.shape[0], -1)
        position_valid = self.position_valid.to(dtype=poses.dtype).reshape(self.positions.shape[0], -1)
        rotation_valid = self.rotation_valid.to(dtype=poses.dtype).reshape(self.positions.shape[0], -1)
        return torch.cat((poses, position_valid, rotation_valid), dim=-1)


class SparsePoseCommandGenerator:
    """Produces sparse human pose commands for training."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dt: float,
        mode: str = "synthetic",
        dataset_path: str | None = None,
        expected_embodiment: str | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mode = mode
        self.num_segments = len(TRACKED_SEGMENTS)

        self.phase = torch.rand(self.num_envs, device=self.device) * (2.0 * torch.pi)
        self.frame_time_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sequence_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sequence_elapsed_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.command_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.dataset = None
        if self.mode == "dataset":
            if not dataset_path:
                raise ValueError("dataset mode requires teleop_dataset_path")
            self.dataset = self._load_dataset(dataset_path, expected_embodiment)
            self.max_frame = int(self.dataset["positions"].shape[0])
            if self.dataset["sequence_starts"] is not None:
                self.num_sequences = int(self.dataset["sequence_starts"].shape[0])
                self.sequence_ids = torch.randint(0, self.num_sequences, (self.num_envs,), device=self.device)
                self.sequence_elapsed_s.zero_()
            else:
                frame_fps = float(self.dataset["sequence_fps"][0].item())
                random_frames = torch.randint(0, self.max_frame, (self.num_envs,), device=self.device)
                self.frame_time_s = random_frames.float() / frame_fps

    def _load_dataset(self, dataset_path: str, expected_embodiment: str | None = None) -> dict[str, torch.Tensor | None]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")

        data = np.load(path)
        dataset_embodiment = _read_npz_scalar_string(data, "embodiment")
        if expected_embodiment is not None and dataset_embodiment is not None:
            expected_key = expected_embodiment.strip().lower()
            if dataset_embodiment != expected_key:
                raise ValueError(
                    f"Dataset embodiment mismatch for {path}: expected {expected_key!r}, found {dataset_embodiment!r}."
                )
        elif expected_embodiment is not None and dataset_embodiment is None:
            raise ValueError(f"Sparse dataset {path} does not declare an embodiment; expected {expected_embodiment!r}.")
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
        if "target_lin_vel_xy" not in data:
            raise ValueError("Dataset mode requires target_lin_vel_xy root-trajectory commands.")
        root_lin_vel_xy_w = torch.as_tensor(data["target_lin_vel_xy"], dtype=torch.float32, device=self.device)
        if root_lin_vel_xy_w.shape != (raw_positions.shape[0], 2):
            raise ValueError(
                f"Expected target_lin_vel_xy with shape {(raw_positions.shape[0], 2)}, "
                f"got {tuple(root_lin_vel_xy_w.shape)}"
            )
        if "rotation_valid" not in data:
            raise ValueError(
                "Dataset mode requires rotation_valid so pelvis-origin positions can be normalized into the pelvis frame."
            )
        rotation_valid = torch.as_tensor(data["rotation_valid"], dtype=torch.bool, device=self.device)
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
        pelvis_quat = orientations[:, pelvis_idx].clone()
        pelvis_quat_inv = quat_conjugate(pelvis_quat).unsqueeze(1).expand(-1, self.num_segments, -1)

        segment_velocities = torch.zeros_like(positions)
        velocity_valid = torch.zeros_like(position_valid)

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
        sequence_fps = self._resolve_sequence_fps(data, sequence_lengths, path)
        if sequence_starts is not None and sequence_lengths is not None:
            if sequence_starts.ndim != 1 or sequence_lengths.ndim != 1 or sequence_starts.shape != sequence_lengths.shape:
                raise ValueError("sequence_starts and sequence_lengths must be matching 1-D arrays.")
            if torch.any(sequence_lengths <= 0):
                raise ValueError("sequence_lengths must be strictly positive.")
            if torch.any(sequence_starts < 0) or torch.any(sequence_starts + sequence_lengths > raw_positions.shape[0]):
                raise ValueError("sequence_starts/sequence_lengths contain ranges outside the sparse dataset.")
            for seq_start, seq_len, seq_fps in zip(
                sequence_starts.tolist(), sequence_lengths.tolist(), sequence_fps.tolist(), strict=False
            ):
                seq_start = int(seq_start)
                seq_len = int(seq_len)
                if seq_len <= 1:
                    continue
                seq_dt = 1.0 / float(seq_fps)
                delta = positions[seq_start + 1 : seq_start + seq_len] - positions[seq_start : seq_start + seq_len - 1]
                segment_velocities[seq_start + 1 : seq_start + seq_len] = quat_apply(
                    pelvis_quat_inv[seq_start + 1 : seq_start + seq_len],
                    delta / seq_dt,
                )
                velocity_valid[seq_start + 1 : seq_start + seq_len] = (
                    position_valid[seq_start + 1 : seq_start + seq_len]
                    & position_valid[seq_start : seq_start + seq_len - 1]
                    & pelvis_rotation_valid[seq_start + 1 : seq_start + seq_len].unsqueeze(-1)
                )
        else:
            if positions.shape[0] > 1:
                seq_dt = 1.0 / float(sequence_fps[0].item())
                delta = positions[1:] - positions[:-1]
                segment_velocities[1:] = quat_apply(pelvis_quat_inv[1:], delta / seq_dt)
                velocity_valid[1:] = position_valid[1:] & position_valid[:-1] & pelvis_rotation_valid[1:].unsqueeze(-1)

        root_lin_vel_w = torch.zeros((raw_positions.shape[0], 3), dtype=torch.float32, device=self.device)
        root_lin_vel_w[:, :2] = torch.nan_to_num(root_lin_vel_xy_w, nan=0.0, posinf=0.0, neginf=0.0)
        root_lin_vel_local = quat_apply(pelvis_quat_inv[:, 0], root_lin_vel_w)[:, :2]
        positions = quat_apply(pelvis_quat_inv, positions)
        positions[:, pelvis_idx] = 0.0
        position_valid = position_valid & pelvis_rotation_valid.unsqueeze(-1)
        position_valid[:, pelvis_idx] = pelvis_position_valid
        orientations[:, pelvis_idx] = identity[:, pelvis_idx]
        rotation_valid[:, pelvis_idx] = False
        segment_velocities[:, pelvis_idx] = 0.0
        velocity_valid[:, pelvis_idx] = False

        return {
            "positions": positions,
            "orientations": orientations,
            "position_valid": position_valid,
            "rotation_valid": rotation_valid,
            "segment_velocities": segment_velocities,
            "velocity_valid": velocity_valid,
            "root_lin_vel_xy": root_lin_vel_local,
            "sequence_starts": sequence_starts,
            "sequence_lengths": sequence_lengths,
            "sequence_fps": sequence_fps,
        }

    def _resolve_sequence_fps(
        self,
        data: np.lib.npyio.NpzFile,
        sequence_lengths: torch.Tensor | None,
        path: Path,
    ) -> torch.Tensor:
        sequence_count = 1 if sequence_lengths is None else int(sequence_lengths.shape[0])
        if "sequence_fps" in data:
            fps_np = np.asarray(data["sequence_fps"], dtype=np.float32)
        elif "effective_fps" in data:
            fps_np = np.asarray(data["effective_fps"], dtype=np.float32)
        else:
            raise ValueError(f"Sparse dataset {path} is missing required FPS metadata: sequence_fps or effective_fps.")

        if fps_np.shape == ():
            fps = np.full(sequence_count, float(fps_np), dtype=np.float32)
        elif fps_np.ndim == 1 and len(fps_np) == sequence_count:
            fps = fps_np.astype(np.float32)
        else:
            raise ValueError(
                f"Sparse dataset {path} FPS metadata must be scalar or length {sequence_count}, "
                f"got shape {fps_np.shape}."
            )
        if not np.isfinite(fps).all() or np.any(fps <= 0.0):
            raise ValueError(f"Sparse dataset {path} contains invalid FPS metadata.")
        return torch.as_tensor(fps, dtype=torch.float32, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self.dataset is None:
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device) * (2.0 * torch.pi)
        self.command_done[env_ids] = False
        if self.dataset is not None:
            if self.dataset["sequence_starts"] is not None:
                self.sequence_ids[env_ids] = torch.randint(0, self.num_sequences, (len(env_ids),), device=self.device)
                self.sequence_elapsed_s[env_ids] = 0.0
            else:
                frame_fps = float(self.dataset["sequence_fps"][0].item())
                random_frames = torch.randint(0, self.max_frame, (len(env_ids),), device=self.device)
                self.frame_time_s[env_ids] = random_frames.float() / frame_fps

    def step(self) -> SparsePoseBatch:
        if self.dataset is not None:
            return self._dataset_batch()
        self.command_done.zero_()
        batch = self._synthetic_batch()
        self.phase = torch.remainder(self.phase + self.dt * 2.5, 2.0 * torch.pi)
        return batch

    def current_batch(self) -> SparsePoseBatch:
        if self.dataset is not None:
            return self._dataset_batch(advance=False)
        return self._synthetic_batch()

    def _dataset_batch(self, advance: bool = True) -> SparsePoseBatch:
        if advance:
            self.command_done.zero_()
        if self.dataset["sequence_starts"] is not None:
            seq_starts = self.dataset["sequence_starts"][self.sequence_ids]
            seq_lengths = self.dataset["sequence_lengths"][self.sequence_ids]
            seq_fps = self.dataset["sequence_fps"][self.sequence_ids]
            frame_pos = torch.clamp(self.sequence_elapsed_s * seq_fps, min=0.0)
            offset0 = torch.minimum(torch.floor(frame_pos).to(dtype=torch.long), seq_lengths - 1)
            offset1 = torch.minimum(offset0 + 1, seq_lengths - 1)
            alpha = torch.clamp(frame_pos - offset0.float(), 0.0, 1.0)
            idx0 = seq_starts + offset0
            idx1 = seq_starts + offset1
            sequence_duration = torch.clamp(seq_lengths.float() / seq_fps, min=self.dt)
            phase = (2.0 * torch.pi) * torch.clamp(self.sequence_elapsed_s / sequence_duration, 0.0, 1.0)
        else:
            frame_fps = self.dataset["sequence_fps"][0]
            frame_pos = torch.remainder(self.frame_time_s * frame_fps, float(self.max_frame))
            idx0 = torch.floor(frame_pos).to(dtype=torch.long)
            idx1 = torch.remainder(idx0 + 1, self.max_frame)
            alpha = torch.clamp(frame_pos - idx0.float(), 0.0, 1.0)
            denom = max(self.max_frame - 1, 1)
            phase = (2.0 * torch.pi) * frame_pos / float(denom)
        alpha_pos = alpha.view(-1, 1, 1)
        alpha_mask = alpha.view(-1, 1) > 1.0e-6
        positions0 = self.dataset["positions"][idx0]
        positions1 = self.dataset["positions"][idx1]
        positions = (1.0 - alpha_pos) * positions0 + alpha_pos * positions1
        orientations0 = self.dataset["orientations"][idx0]
        orientations1 = self.dataset["orientations"][idx1]
        quat_sign = torch.where(torch.sum(orientations0 * orientations1, dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
        orientations = _normalize_quat((1.0 - alpha_pos) * orientations0 + alpha_pos * orientations1 * quat_sign)
        position_valid0 = self.dataset["position_valid"][idx0]
        position_valid1 = self.dataset["position_valid"][idx1]
        rotation_valid0 = self.dataset["rotation_valid"][idx0]
        rotation_valid1 = self.dataset["rotation_valid"][idx1]
        velocity_valid0 = self.dataset["velocity_valid"][idx0]
        velocity_valid1 = self.dataset["velocity_valid"][idx1]
        position_valid = torch.where(alpha_mask, position_valid0 & position_valid1, position_valid0)
        rotation_valid = torch.where(alpha_mask, rotation_valid0 & rotation_valid1, rotation_valid0)
        velocity_valid = torch.where(alpha_mask, velocity_valid0 & velocity_valid1, velocity_valid0)
        segment_velocities0 = self.dataset["segment_velocities"][idx0]
        segment_velocities1 = self.dataset["segment_velocities"][idx1]
        segment_velocities = (1.0 - alpha_pos) * segment_velocities0 + alpha_pos * segment_velocities1
        alpha_root = alpha.view(-1, 1)
        root_lin_vel_xy0 = self.dataset["root_lin_vel_xy"][idx0]
        root_lin_vel_xy1 = self.dataset["root_lin_vel_xy"][idx1]
        root_lin_vel_xy = (1.0 - alpha_root) * root_lin_vel_xy0 + alpha_root * root_lin_vel_xy1
        if advance and self.dataset["sequence_starts"] is not None:
            next_elapsed = self.sequence_elapsed_s + self.dt
            self.command_done.copy_(next_elapsed * seq_fps >= seq_lengths.float())
            sequence_stop_time = seq_lengths.float() / seq_fps
            self.sequence_elapsed_s = torch.minimum(next_elapsed, sequence_stop_time)
        elif advance:
            frame_fps = self.dataset["sequence_fps"][0]
            sequence_duration = float(self.max_frame) / float(frame_fps.item())
            self.frame_time_s = torch.remainder(self.frame_time_s + self.dt, sequence_duration)
        return SparsePoseBatch(
            positions,
            orientations,
            position_valid,
            rotation_valid,
            phase,
            segment_velocities=segment_velocities,
            velocity_valid=velocity_valid,
            root_lin_vel_xy=root_lin_vel_xy,
        )

    def _synthetic_batch(self) -> SparsePoseBatch:
        batch = self.num_envs
        positions = torch.zeros(batch, self.num_segments, 3, dtype=torch.float32, device=self.device)
        orientations = torch.zeros(batch, self.num_segments, 4, dtype=torch.float32, device=self.device)
        position_valid = torch.ones(batch, self.num_segments, dtype=torch.bool, device=self.device)
        rotation_valid = torch.ones(batch, self.num_segments, dtype=torch.bool, device=self.device)
        orientations[..., 3] = 1.0

        sway = 0.03 * torch.sin(self.phase)
        shoulder = 0.16 + 0.02 * torch.sin(self.phase + 0.4)
        arm_reach = 0.16 + 0.04 * torch.cos(self.phase)
        knee = -0.13 + 0.02 * torch.cos(self.phase)
        foot_x = 0.05 * torch.sin(self.phase)
        foot_z = -0.28 + 0.02 * torch.sin(self.phase * 2.0)

        positions[:, SEGMENT_INDEX["pelvis"]] = torch.tensor((0.0, 0.0, 0.0), device=self.device)
        positions[:, SEGMENT_INDEX["head"]] = torch.stack(
            (0.06 * torch.cos(self.phase), sway, torch.full_like(sway, 0.33)), dim=-1
        )
        positions[:, SEGMENT_INDEX["left_hand"]] = torch.stack(
            (arm_reach, shoulder, 0.14 + 0.05 * torch.sin(self.phase + 0.7)), dim=-1
        )
        positions[:, SEGMENT_INDEX["right_hand"]] = torch.stack(
            (arm_reach, -shoulder, 0.14 + 0.05 * torch.sin(self.phase + 2.2)), dim=-1
        )
        positions[:, SEGMENT_INDEX["left_knee"]] = torch.stack(
            (0.02 + 0.02 * torch.cos(self.phase), 0.05 + 0.01 * torch.sin(self.phase), knee), dim=-1
        )
        positions[:, SEGMENT_INDEX["right_knee"]] = torch.stack(
            (0.02 + 0.02 * torch.cos(self.phase), -0.05 - 0.01 * torch.sin(self.phase), knee), dim=-1
        )
        positions[:, SEGMENT_INDEX["left_foot"]] = torch.stack(
            (
                foot_x,
                torch.full_like(sway, 0.06),
                foot_z,
            ),
            dim=-1,
        )
        positions[:, SEGMENT_INDEX["right_foot"]] = torch.stack(
            (
                -foot_x,
                torch.full_like(sway, -0.06),
                foot_z,
            ),
            dim=-1,
        )

        rotation_valid[:, SEGMENT_INDEX["pelvis"]] = False
        position_valid[:, SEGMENT_INDEX["pelvis"]] = True
        root_lin_vel_xy = torch.zeros(batch, 2, dtype=torch.float32, device=self.device)
        return SparsePoseBatch(positions, orientations, position_valid, rotation_valid, self.phase.clone(), root_lin_vel_xy=root_lin_vel_xy)
