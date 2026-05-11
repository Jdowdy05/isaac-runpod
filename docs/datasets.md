# Dataset Notes

## Current Downloadable Starter Set

These are the datasets wired into the unattended download script:

- `AIST++` annotations:
  - motion data
  - 3D keypoints
  - camera metadata
- `AMASS_Retargeted_for_G1` on Hugging Face:
  - optional robot-motion prior
  - useful as a locomotion and motion-style prior
  - not a substitute for raw human motion data

## Why This Is Only a Starter Set

The current open bundle is enough to:

- bring up the data pipeline
- sanity-check sparse-pose preprocessing
- start early policy and reward tuning
- bootstrap locomotion priors

It is not enough for a strong final teleoperation dataset because:

- AIST++ is dance-focused rather than XR teleoperation-focused.
- AIST++ gives strong body and limb positions, but not the headset-controller-style sparse observations you ultimately care about.
- AIST++ does not provide the full hand-pose richness you likely want for later whole-body teleoperation.
- The retargeted AMASS derivative is robot-motion data, not the original human dataset.

## Planned Later Additions

The next data additions should be:

1. Licensed `AMASS`, especially for broad full-body motion coverage.
2. A headset-centric dataset such as `EgoBody`, if you want stronger alignment with VR teleoperation.
3. Your own teleoperation logs once the interface stabilizes around headset/controllers or Vive trackers.

## Practical Recommendation

Use the open bundle now to make the codepath real. Do not treat it as the final training corpus.

## Canonical Processed Paths

OP3 and G1 sparse datasets are embodiment-specific and should not be swapped.

- OP3/open: `data/processed/open/aist_sparse_pose.npz`
- OP3/open: `data/processed/open/amass_sparse_pose.npz`
- OP3/open: `data/processed/open/teleop_sparse_pose.npz`
- G1: `data/processed/g1/aist_sparse_pose.npz`
- G1: `data/processed/g1/amass_sparse_pose.npz`
- G1: `data/processed/g1/teleop_sparse_pose.npz`

Before launching G1 training or playback, confirm the selected dataset path starts with `data/processed/g1/`.

## Required Sparse NPZ Metadata

Current sparse datasets must declare:

- `embodiment`: scalar string such as `op3` or `g1`
- `sequence_fps`: one FPS value per sequence, or scalar `effective_fps` for single-rate legacy-compatible files
- `target_lin_vel_xy`: root XY velocity command before runtime conversion into the pelvis/root frame
- `sequence_starts` and `sequence_lengths` for clip boundaries
- `source_datasets` and `sequence_source_dataset` for provenance after merge/filter

Merging now rejects mixed embodiments, and runtime loading rejects datasets that do not declare the expected embodiment. Older NPZs that lack these fields should be regenerated.

For G1, AIST and AMASS are now preprocessed directly with the G1 embodiment profile. The sparse slot named `head` is generated as an upper-torso/neck surrogate for G1 because the current G1 asset tracks that slot with `torso_link`.

RunPod rebuild note from 2026-05-09: the current final G1 training file at `/workspace/isaac-runpod/data/processed/g1/teleop_sparse_pose.npz` is direct-G1 AMASS-derived (`8,253` clips, `1,885,671` frames). AIST was regenerated but rejected by the current G1 feasibility filter, so do not assume the final file contains AIST clips just because the preparation script builds AIST first.

## Local Raw Data Staging

Local raw data is gitignored under `data/raw/`. As of 2026-05-11:

- `data/raw/archive/amass_smplh_g/` contains one canonical local copy of the AMASS SMPL+H G subset archives from `/Users/jordan/Downloads/smplh`.
- `/Users/jordan/Downloads/smplh_2` was checked and matched the same archive set byte-for-byte, so it is treated as a duplicate.
- `data/raw/archive/smplh_models/` contains the actual SMPL-H model PKLs copied from `/Users/jordan/Downloads/smplx/smplh`.
- `data/raw/smplh/` contains a script-ready local copy of `SMPLH_MALE.pkl` and `SMPLH_FEMALE.pkl`.
- Local AIST++ was not found; the active RunPod still has `data/raw/aistplusplus`.

Despite the folder names, `/Users/jordan/Downloads/smplh` and `/Users/jordan/Downloads/smplh_2` are AMASS archive folders, not SMPL-H model folders. The actual model files used by `prepare_amass_sparse.py` came from `/Users/jordan/Downloads/smplx/smplh`.
