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
