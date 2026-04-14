# OP3 Teleop Lab

External Isaac Lab project scaffold for end-to-end teleoperation policy learning on the Robotis OP3. The intended policy maps sparse human observations and robot state directly to OP3 joint position targets, with locomotion and balance included in the task.

The project is structured to work with Isaac Lab's external-project pattern and keeps two physics paths:

- Newton as the primary training backend.
- PhysX as a fallback path for compatibility and debugging.
- The OP3 task timing is fixed to a 0.002 s physics step with decimation 10, so the policy runs at 50 Hz.

The current repository is a scaffold, not a finished training system. It gives you:

- An Isaac Lab extension package with direct-workflow task registration.
- An OP3 teleop locomotion environment skeleton with joint-position actions.
- A sparse human observation schema with sensor validity masks.
- RunPod bootstrap scripts.
- Open-dataset download and preprocessing scripts.
- Notes on the later addition of licensed motion sources such as AMASS.
- A standalone ADD trainer path based on the MimicKit paper and codebase.

## Project Layout

```text
.
├── README.md
├── scripts
│   ├── data
│   ├── list_envs.py
│   ├── rl_games
│   └── runpod
└── source
    └── op3_teleop_lab
        ├── config
        ├── docs
        ├── op3_teleop_lab
        └── setup.py
```

## Core Assumptions

- The OP3 asset will later exist as an Isaac Lab asset config named `OP3_CFG`.
- The first code change after adding the OP3 asset will usually be updating
  `source/op3_teleop_lab/op3_teleop_lab/tasks/direct/op3_teleop/robot_profile.py`
  so the joint and body names match the imported asset exactly.
- Open datasets are enough to build the pipeline, sanity-check preprocessing,
  and start early policy work, but they are not a strong final answer for
  high-quality full-body teleoperation. Plan to add licensed motion data later,
  especially AMASS.

## Getting Started

1. Clone or mount this project into your RunPod workspace.
2. Run `scripts/runpod/bootstrap.sh`.
3. Run `scripts/runpod/download_open_datasets.sh`.
4. Install the extension into Isaac Lab:

   ```bash
   python -m pip install -e source/op3_teleop_lab
   ```

5. List environments:

   ```bash
   python scripts/list_envs.py
   ```

6. Train with RL Games once Isaac Lab is installed:

   ```bash
   python scripts/rl_games/train.py --task Isaac-OP3-Teleop-Direct-v0 --headless
   ```

For Newton, use:

```bash
python scripts/rl_games/train.py --task Isaac-OP3-Teleop-Newton-Direct-v0 --headless
```

To train with ADD instead of the RL Games baseline:

```bash
python scripts/add/train.py --task Isaac-OP3-Teleop-Newton-Direct-v0 --headless
```

## Dataset Strategy

`scripts/runpod/download_open_datasets.sh` downloads only sources that can be fetched unattended. That starter bundle is intentionally conservative.

The longer-term plan is:

- Start with open, unattended datasets for pipeline bring-up.
- Add stronger licensed motion data later, especially AMASS.
- Optionally add EgoBody or similar XR-centric data later if you want tighter alignment with headset-and-controller teleoperation.
