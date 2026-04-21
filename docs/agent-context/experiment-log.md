# Experiment Log

Use this file for durable experiment history.

## Entry Template

- Date:
- Goal:
- Setup:
- Result:
- Interpretation:
- Next action:

## Entries

- Date: 2026-04-15
- Goal: Set up reusable agent skills and persistent context files.
- Setup: Installed official productivity skills and created custom local skills plus notebook templates.
- Result: Environment is prepared for future robotics, RL, IL, coding, and paper-analysis work.
- Interpretation: Cross-session context now has a stable home in the repository.
- Next action: Add the first real experiment entry when model, simulator, or training work begins.

- Date: 2026-04-16
- Goal: Isolate whether the OP3 floor-penetration issue comes from the Newton environment setup or from playback/recording.
- Setup: Added `scripts/debug/test_newton_ground_contact.py` plus `scripts/runpod/test_newton_ground_contact.sh` to run a single-robot Newton drop test with the same OP3 asset and plane terrain configuration used by the teleoperation environment.
- Result: The repository now has a standalone diagnostic that prints ground-prim validity, collision prim discovery, root pose, and foot heights while stepping one OP3 instance until completion or a root height below `-5.0`.
- Interpretation: This creates a clean first-line test before changing terrain, contact, or asset settings inside the main RL environment.
- Next action: Run the diagnostic on RunPod and use its output to decide whether the next fix belongs in terrain creation, the OP3 USD/collision setup, or the recording/playback pipeline.

- Date: 2026-04-16
- Goal: Confirm whether Newton floor penetration is a terrain problem or an OP3/Newton collision compatibility problem, and choose a safe backend fallback.
- Setup: Pulled the standalone diagnostic to RunPod and ran `bash scripts/runpod/test_newton_ground_contact.sh --headless --steps 600 --print_every 20`.
- Result: The diagnostic reported a valid ground plane with collision prim `/World/ground/terrain/GroundPlane/CollisionPlane`, but the OP3 root height still dropped monotonically from `0.285` to below `-5.0` in the minimal one-robot Newton test.
- Interpretation: The current failure is not caused by the camera recorder and not by a missing plane; it is most likely a Newton-versus-OP3-collision issue in the asset or contact representation.
- Next action: Use PhysX as the default backend for OP3 training/playback for now, and investigate simplifying or reauthoring OP3 collision geometry before returning to Newton.

- Date: 2026-04-16
- Goal: Determine whether the PhysX playback path is disconnected from the actor or whether the policy mean is simply near zero.
- Setup: Added `--sample_actions` and `--print_stats_every` to `scripts/add/play.py` and `scripts/add/record_camera_playback.py`, then ran a one-iteration PhysX debug training job on RunPod and replayed the resulting checkpoint in both deterministic and sampled modes.
- Result: Deterministic playback received nonzero observations and produced nonzero actions, but the action mean stayed very small (roughly `0.003–0.016` in normalized action units per joint). Sampled playback from the same checkpoint produced much larger actions (roughly `0.03–0.12`), confirming that action flow is working and that deterministic playback can appear nearly motionless when the learned mean remains close to zero.
- Interpretation: The current “robot does nothing” symptom under PhysX is not caused by broken observation delivery, action delivery, or joint-name ordering. It is consistent with weak learning plus deterministic playback using the policy mean while training uses stochastic sampling.
- Next action: Use the new playback diagnostics on later checkpoints, and if deterministic means remain near zero after substantial training, revisit the optimization setup rather than the action plumbing.

- Date: 2026-04-21
- Goal: Rebuild a clean RunPod environment and restart OP3 full-body teleoperation training using PhysX.
- Setup: Cloned `Jdowdy05/isaac-runpod` onto new RunPod `knzldgd9e02d4b`, ran `scripts/runpod/bootstrap.sh`, transferred local SMPL-H model files and AMASS `SMPL+H G` subset archives with `runpodctl`, extracted AMASS, downloaded/preprocessed AIST++, ran `scripts/runpod/prepare_amass_dataset.sh`, and launched `scripts/runpod/train_add_physx.sh`.
- Result: The merged dataset was rebuilt at `/workspace/isaac-runpod/data/processed/open/teleop_sparse_pose.npz` with `1,202,427` frames (`234,967` AIST frames plus `967,460` AMASS frames). PhysX training started from `logs/train_2026-04-21_05-34-00.log` with main Python PID `2999`.
- Interpretation: The new pod is training-ready and uses the PhysX backend, avoiding the known Newton ground-contact issue. The training log may be buffered under `nohup`, so process/GPU checks are more reliable than tailing the log during early iterations.
- Next action: Monitor the first metrics/checkpoint in `checkpoints/add/2026-04-21` and evaluate playback once a meaningful checkpoint is available.

- Date: 2026-04-21
- Goal: Address checkpoint `5000` playback showing OP3 barely moving.
- Setup: Stopped the previous PhysX ADD run, increased ADD teacher exploration from `0.02` fixed std to a schedule starting at `1.5` and decaying to `0.25` over `50000` iterations, added `sampled_action_abs_mean/max` logging, and tried larger environment counts based on low VRAM usage.
- Result: `4096` and `3072` environments both stalled in Isaac startup before the first metric line, so the stable default was restored to `2048`. A new run started from `logs/train_2026-04-21_16-43-41.log` with checkpoint directory `/workspace/isaac-runpod/checkpoints/add/2026-04-21_04`; early metrics showed `sampled_action_abs_mean` about `1.19` with deterministic teacher/student means near `0.015`.
- Interpretation: The motionless playback was consistent with near-zero deterministic policy means and very low prior exploration. The new run now explores substantially larger normalized actions while keeping the known-good `2048` environment count.
- Next action: Let the run reach an early checkpoint, then replay deterministic and sampled teacher/student behavior to see whether the policy mean grows beyond the prior `0.06-0.07` range.
