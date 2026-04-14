#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


DEFAULT_SUBSETS = (
    "ACCAD",
    "BMLmovi",
    "BMLrub",
    "CMU",
    "EKUT",
    "EyesJapanDataset",
    "HDM05",
    "HumanEva",
    "KIT",
    "TotalCapture",
    "Transitions",
    "DanceDB",
)

SUBSET_ALIASES = {
    "ACCAD": ("ACCAD",),
    "BMLhandball": ("BMLhandball",),
    "BMLmovi": ("BMLmovi",),
    "BMLrub": ("BMLrub", "BioMotionLab_NTroje"),
    "CMU": ("CMU",),
    "CNRS": ("CNRS",),
    "DanceDB": ("DanceDB",),
    "DFaust": ("DFaust", "DFaust_67"),
    "EKUT": ("EKUT",),
    "EyesJapanDataset": ("EyesJapanDataset", "Eyes_Japan_Dataset"),
    "GRAB": ("GRAB",),
    "HDM05": ("HDM05", "MPI_HDM05"),
    "HUMAN4D": ("HUMAN4D",),
    "HumanEva": ("HumanEva",),
    "KIT": ("KIT",),
    "MoSh": ("MoSh", "MPI_mosh"),
    "PosePrior": ("PosePrior",),
    "SFU": ("SFU",),
    "SOMA": ("SOMA",),
    "SSM": ("SSM", "SSM_synced"),
    "TCDHands": ("TCDHands", "TCD_handMocap"),
    "TotalCapture": ("TotalCapture",),
    "Transitions": ("Transitions", "Transitions_mocap"),
    "WEIZMANN": ("WEIZMANN",),
}

SMPLH_BODY_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

SEGMENTS = (
    "pelvis",
    "head",
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)
SEGMENT_INDEX = {name: idx for idx, name in enumerate(SEGMENTS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AMASS SMPL+H motion files into sparse teleoperation commands.")
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--subsets", nargs="*", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def decode_gender(value) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip().lower()
    if value in {"male", "female", "neutral"}:
        return value
    return "neutral"


def finite_mask(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def forward_fill(points: np.ndarray) -> np.ndarray:
    output = points.copy()
    valid = np.isfinite(output).all(axis=-1)
    if not np.any(valid):
        return np.zeros_like(output)
    first_valid = int(np.argmax(valid))
    output[:first_valid] = output[first_valid]
    for idx in range(first_valid + 1, len(output)):
        if not valid[idx]:
            output[idx] = output[idx - 1]
    return output


def normalize_vectors(vectors: np.ndarray, eps: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    valid = norms[..., 0] > eps
    normalized = np.zeros_like(vectors, dtype=np.float32)
    normalized[valid] = vectors[valid] / norms[valid]
    return normalized, valid


def make_frame_from_forward_up(forward_hint: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_axis, x_valid = normalize_vectors(forward_hint)
    y_axis, y_valid = normalize_vectors(np.cross(up_hint, x_axis))
    z_axis, z_valid = normalize_vectors(np.cross(x_axis, y_axis))
    mats = np.stack((x_axis, y_axis, z_axis), axis=-1).astype(np.float32)
    valid = x_valid & y_valid & z_valid
    mats[~valid] = np.eye(3, dtype=np.float32)
    return mats, valid


def rotation_matrices_to_quats_xyzw(mats: np.ndarray) -> np.ndarray:
    quats = np.zeros((mats.shape[0], 4), dtype=np.float32)
    trace = mats[:, 0, 0] + mats[:, 1, 1] + mats[:, 2, 2]

    positive = trace > 0.0
    if np.any(positive):
        s = np.sqrt(trace[positive] + 1.0) * 2.0
        quats[positive, 0] = (mats[positive, 2, 1] - mats[positive, 1, 2]) / s
        quats[positive, 1] = (mats[positive, 0, 2] - mats[positive, 2, 0]) / s
        quats[positive, 2] = (mats[positive, 1, 0] - mats[positive, 0, 1]) / s
        quats[positive, 3] = 0.25 * s

    cond_x = (~positive) & (mats[:, 0, 0] > mats[:, 1, 1]) & (mats[:, 0, 0] > mats[:, 2, 2])
    if np.any(cond_x):
        s = np.sqrt(1.0 + mats[cond_x, 0, 0] - mats[cond_x, 1, 1] - mats[cond_x, 2, 2]) * 2.0
        quats[cond_x, 0] = 0.25 * s
        quats[cond_x, 1] = (mats[cond_x, 0, 1] + mats[cond_x, 1, 0]) / s
        quats[cond_x, 2] = (mats[cond_x, 0, 2] + mats[cond_x, 2, 0]) / s
        quats[cond_x, 3] = (mats[cond_x, 2, 1] - mats[cond_x, 1, 2]) / s

    cond_y = (~positive) & (~cond_x) & (mats[:, 1, 1] > mats[:, 2, 2])
    if np.any(cond_y):
        s = np.sqrt(1.0 + mats[cond_y, 1, 1] - mats[cond_y, 0, 0] - mats[cond_y, 2, 2]) * 2.0
        quats[cond_y, 0] = (mats[cond_y, 0, 1] + mats[cond_y, 1, 0]) / s
        quats[cond_y, 1] = 0.25 * s
        quats[cond_y, 2] = (mats[cond_y, 1, 2] + mats[cond_y, 2, 1]) / s
        quats[cond_y, 3] = (mats[cond_y, 0, 2] - mats[cond_y, 2, 0]) / s

    cond_z = (~positive) & (~cond_x) & (~cond_y)
    if np.any(cond_z):
        s = np.sqrt(1.0 + mats[cond_z, 2, 2] - mats[cond_z, 0, 0] - mats[cond_z, 1, 1]) * 2.0
        quats[cond_z, 0] = (mats[cond_z, 0, 2] + mats[cond_z, 2, 0]) / s
        quats[cond_z, 1] = (mats[cond_z, 1, 2] + mats[cond_z, 2, 1]) / s
        quats[cond_z, 2] = 0.25 * s
        quats[cond_z, 3] = (mats[cond_z, 1, 0] - mats[cond_z, 0, 1]) / s

    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats /= np.clip(norms, 1.0e-6, None)
    return quats


def build_sparse_sequence_from_joints(joints: np.ndarray, effective_fps: float) -> tuple[np.ndarray, ...]:
    num_frames = joints.shape[0]
    positions = np.zeros((num_frames, len(SEGMENTS), 3), dtype=np.float32)
    orientations = np.zeros((num_frames, len(SEGMENTS), 4), dtype=np.float32)
    orientations[..., 3] = 1.0
    position_valid = np.ones((num_frames, len(SEGMENTS)), dtype=bool)
    rotation_valid = np.zeros((num_frames, len(SEGMENTS)), dtype=bool)

    pelvis = joints[:, SMPLH_BODY_JOINTS["pelvis"]]
    head = joints[:, SMPLH_BODY_JOINTS["head"]]
    neck = joints[:, SMPLH_BODY_JOINTS["neck"]]
    left_shoulder = joints[:, SMPLH_BODY_JOINTS["left_shoulder"]]
    right_shoulder = joints[:, SMPLH_BODY_JOINTS["right_shoulder"]]
    left_elbow = joints[:, SMPLH_BODY_JOINTS["left_elbow"]]
    right_elbow = joints[:, SMPLH_BODY_JOINTS["right_elbow"]]
    left_wrist = joints[:, SMPLH_BODY_JOINTS["left_wrist"]]
    right_wrist = joints[:, SMPLH_BODY_JOINTS["right_wrist"]]
    left_hip = joints[:, SMPLH_BODY_JOINTS["left_hip"]]
    right_hip = joints[:, SMPLH_BODY_JOINTS["right_hip"]]
    left_knee = joints[:, SMPLH_BODY_JOINTS["left_knee"]]
    right_knee = joints[:, SMPLH_BODY_JOINTS["right_knee"]]
    left_ankle = joints[:, SMPLH_BODY_JOINTS["left_ankle"]]
    right_ankle = joints[:, SMPLH_BODY_JOINTS["right_ankle"]]
    left_foot = joints[:, SMPLH_BODY_JOINTS["left_foot"]]
    right_foot = joints[:, SMPLH_BODY_JOINTS["right_foot"]]

    raw_targets = {
        "pelvis": pelvis,
        "head": head,
        "left_hand": left_wrist,
        "right_hand": right_wrist,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_foot": left_ankle,
        "right_foot": right_ankle,
    }

    pelvis_filled = forward_fill(pelvis)
    for segment_name, points in raw_targets.items():
        seg_idx = SEGMENT_INDEX[segment_name]
        positions[:, seg_idx] = points - pelvis_filled
        position_valid[:, seg_idx] = finite_mask(points)

    positions[:, SEGMENT_INDEX["pelvis"]] = 0.0
    target_lin_vel_xy = np.diff(pelvis_filled[:, :2], axis=0, prepend=pelvis_filled[:1, :2]) * effective_fps
    target_lin_vel_xy = target_lin_vel_xy.astype(np.float32)

    pelvis_lateral = (left_hip - right_hip) + (left_shoulder - right_shoulder)
    pelvis_up_hint = 0.5 * (left_shoulder + right_shoulder) - pelvis
    pelvis_forward_hint = np.cross(pelvis_lateral, pelvis_up_hint)
    pelvis_world, pelvis_rot_valid = make_frame_from_forward_up(pelvis_forward_hint, pelvis_up_hint)
    pelvis_world_inv = np.swapaxes(pelvis_world, -1, -2)
    pelvis_forward = pelvis_world[:, :, 0]
    pelvis_up = pelvis_world[:, :, 2]

    orientations[:, SEGMENT_INDEX["pelvis"]] = rotation_matrices_to_quats_xyzw(pelvis_world)
    rotation_valid[:, SEGMENT_INDEX["pelvis"]] = pelvis_rot_valid

    def set_relative_orientation(segment_name: str, world_mat: np.ndarray, valid_mask: np.ndarray) -> None:
        seg_idx = SEGMENT_INDEX[segment_name]
        relative_mat = np.einsum("tij,tjk->tik", pelvis_world_inv, world_mat).astype(np.float32)
        orientations[:, seg_idx] = rotation_matrices_to_quats_xyzw(relative_mat)
        rotation_valid[:, seg_idx] = valid_mask & pelvis_rot_valid

    head_world, head_rot_valid = make_frame_from_forward_up(
        np.cross(left_shoulder - right_shoulder, head - neck),
        head - neck,
    )
    set_relative_orientation("head", head_world, head_rot_valid)

    left_hand_world, left_hand_rot_valid = make_frame_from_forward_up(left_wrist - left_elbow, pelvis_up)
    set_relative_orientation("left_hand", left_hand_world, left_hand_rot_valid)

    right_hand_world, right_hand_rot_valid = make_frame_from_forward_up(right_wrist - right_elbow, pelvis_up)
    set_relative_orientation("right_hand", right_hand_world, right_hand_rot_valid)

    left_knee_world, left_knee_rot_valid = make_frame_from_forward_up(left_ankle - left_knee, pelvis_up)
    set_relative_orientation("left_knee", left_knee_world, left_knee_rot_valid)

    right_knee_world, right_knee_rot_valid = make_frame_from_forward_up(right_ankle - right_knee, pelvis_up)
    set_relative_orientation("right_knee", right_knee_world, right_knee_rot_valid)

    left_foot_world, left_foot_rot_valid = make_frame_from_forward_up(left_foot - left_ankle, left_knee - left_ankle)
    set_relative_orientation("left_foot", left_foot_world, left_foot_rot_valid)

    right_foot_world, right_foot_rot_valid = make_frame_from_forward_up(right_foot - right_ankle, right_knee - right_ankle)
    set_relative_orientation("right_foot", right_foot_world, right_foot_rot_valid)

    pelvis_rot_missing = ~rotation_valid[:, SEGMENT_INDEX["pelvis"]]
    if np.any(pelvis_rot_missing):
        fallback_world, fallback_valid = make_frame_from_forward_up(pelvis_forward, pelvis_up)
        orientations[pelvis_rot_missing, SEGMENT_INDEX["pelvis"]] = rotation_matrices_to_quats_xyzw(
            fallback_world[pelvis_rot_missing]
        )
        rotation_valid[pelvis_rot_missing, SEGMENT_INDEX["pelvis"]] = fallback_valid[pelvis_rot_missing]

    return positions, orientations, position_valid, rotation_valid, target_lin_vel_xy


def load_motion_arrays(data: np.lib.npyio.NpzFile, target_fps: float) -> tuple[np.ndarray, ...]:
    if "mocap_framerate" in data:
        fps = float(data["mocap_framerate"])
    elif "mocap_frame_rate" in data:
        fps = float(data["mocap_frame_rate"])
    else:
        fps = 60.0
    stride = max(1, int(round(fps / target_fps)))

    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)[::stride]
        trans = np.asarray(data["trans"], dtype=np.float32)[::stride]
        if poses.shape[-1] < 66:
            raise ValueError(f"Unsupported AMASS pose width: {poses.shape[-1]}")
        root_orient = poses[:, :3]
        body_pose = poses[:, 3:66]
        left_hand_pose = np.zeros((len(poses), 45), dtype=np.float32)
        right_hand_pose = np.zeros((len(poses), 45), dtype=np.float32)
        if poses.shape[-1] >= 156:
            left_hand_pose = poses[:, 66:111]
            right_hand_pose = poses[:, 111:156]
    else:
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)[::stride]
        body_pose = np.asarray(data["pose_body"], dtype=np.float32)[::stride]
        trans = np.asarray(data["trans"], dtype=np.float32)[::stride]
        if "pose_hand" in data:
            pose_hand = np.asarray(data["pose_hand"], dtype=np.float32)[::stride]
            left_hand_pose = pose_hand[:, :45]
            right_hand_pose = pose_hand[:, 45:90]
        else:
            left_hand_pose = np.zeros((len(root_orient), 45), dtype=np.float32)
            right_hand_pose = np.zeros((len(root_orient), 45), dtype=np.float32)

    effective_fps = fps / float(stride)
    return root_orient, body_pose, left_hand_pose, right_hand_pose, trans, effective_fps


def resolve_model_ext(model_root: Path) -> str:
    if any(model_root.rglob("*.pkl")):
        return "pkl"
    if any(model_root.rglob("*.npz")):
        return "npz"
    raise FileNotFoundError(
        f"Could not find SMPL/SMPL-H model files under {model_root}. Download the body models first."
    )


def build_model_cache(model_root: Path, model_ext: str, device: torch.device):
    try:
        import smplx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "prepare_amass_sparse.py requires the 'smplx' package. Re-run scripts/runpod/bootstrap.sh after pulling."
        ) from exc

    cache: dict[str, object] = {}

    def get_model(gender: str):
        if gender not in cache:
            cache[gender] = smplx.create(
                str(model_root),
                model_type="smplh",
                gender=gender,
                ext=model_ext,
                use_pca=False,
                flat_hand_mean=True,
                num_betas=16,
            ).to(device)
        return cache[gender]

    return get_model


def iter_motion_files(amass_root: Path, subsets: list[str]) -> list[Path]:
    subset_paths: list[Path] = []
    for subset in subsets:
        aliases = SUBSET_ALIASES.get(subset, (subset,))
        for alias in aliases:
            candidate = amass_root / alias
            if candidate.exists():
                subset_paths.append(candidate)
                break
    motion_paths: list[Path] = []
    for subset_path in subset_paths:
        motion_paths.extend(sorted(path for path in subset_path.rglob("*.npz") if path.name != "shape.npz"))
    return motion_paths


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ext = resolve_model_ext(args.smpl_model_root)
    get_model = build_model_cache(args.smpl_model_root, model_ext, device)

    position_blocks: list[np.ndarray] = []
    orientation_blocks: list[np.ndarray] = []
    position_valid_blocks: list[np.ndarray] = []
    rotation_valid_blocks: list[np.ndarray] = []
    velocity_blocks: list[np.ndarray] = []
    sequence_starts: list[int] = []
    sequence_lengths: list[int] = []
    sources: list[str] = []
    total_frames = 0

    motion_paths = iter_motion_files(args.amass_root, args.subsets)
    if args.limit is not None:
        motion_paths = motion_paths[: args.limit]
    if not motion_paths:
        raise RuntimeError(f"No AMASS motion files found under {args.amass_root} for subsets {args.subsets}")

    for motion_path in motion_paths:
        data = np.load(motion_path, allow_pickle=True)
        gender = decode_gender(data["gender"] if "gender" in data else "neutral")
        betas = np.asarray(data["betas"], dtype=np.float32)
        if betas.shape[0] < 16:
            betas = np.pad(betas, (0, 16 - betas.shape[0]))
        betas = betas[:16]
        root_orient, body_pose, left_hand_pose, right_hand_pose, trans, effective_fps = load_motion_arrays(
            data,
            target_fps=args.target_fps,
        )

        if len(root_orient) < args.min_frames:
            continue

        model = get_model(gender)
        joints_batches: list[np.ndarray] = []
        betas_batch = np.broadcast_to(betas, (min(args.chunk_size, len(root_orient)), len(betas))).copy()

        with torch.no_grad():
            for start in range(0, len(root_orient), args.chunk_size):
                end = min(start + args.chunk_size, len(root_orient))
                batch_size = end - start
                joints = model(
                    global_orient=torch.as_tensor(root_orient[start:end], dtype=torch.float32, device=device),
                    body_pose=torch.as_tensor(body_pose[start:end], dtype=torch.float32, device=device),
                    left_hand_pose=torch.as_tensor(left_hand_pose[start:end], dtype=torch.float32, device=device),
                    right_hand_pose=torch.as_tensor(right_hand_pose[start:end], dtype=torch.float32, device=device),
                    transl=torch.as_tensor(trans[start:end], dtype=torch.float32, device=device),
                    betas=torch.as_tensor(betas_batch[:batch_size], dtype=torch.float32, device=device),
                ).joints[:, :22]
                joints_batches.append(joints.cpu().numpy().astype(np.float32))

        joints_np = np.concatenate(joints_batches, axis=0)
        positions, orientations, position_valid, rotation_valid, target_lin_vel_xy = build_sparse_sequence_from_joints(
            joints_np,
            effective_fps=effective_fps,
        )

        position_blocks.append(positions)
        orientation_blocks.append(orientations)
        position_valid_blocks.append(position_valid)
        rotation_valid_blocks.append(rotation_valid)
        velocity_blocks.append(target_lin_vel_xy)
        sequence_starts.append(total_frames)
        sequence_lengths.append(len(positions))
        sources.append(str(motion_path))
        total_frames += len(positions)

    if not position_blocks:
        raise RuntimeError("No usable AMASS motion files were converted into sparse teleop commands.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=np.concatenate(position_blocks, axis=0),
        orientations=np.concatenate(orientation_blocks, axis=0),
        position_valid=np.concatenate(position_valid_blocks, axis=0),
        rotation_valid=np.concatenate(rotation_valid_blocks, axis=0),
        target_lin_vel_xy=np.concatenate(velocity_blocks, axis=0),
        sequence_starts=np.asarray(sequence_starts, dtype=np.int64),
        sequence_lengths=np.asarray(sequence_lengths, dtype=np.int64),
        segment_names=np.asarray(SEGMENTS),
        source=np.asarray(sources, dtype=str),
        effective_fps=float(args.target_fps),
    )
    print(f"Wrote AMASS sparse dataset to: {args.output}")
    print(f"Sequences: {len(sequence_lengths)}")
    print(f"Frames: {total_frames}")


if __name__ == "__main__":
    main()
