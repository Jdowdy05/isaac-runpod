# Changelog

## Unreleased

- Clarified that OP3 and G1 require separate processed sparse datasets.
- Clarified that ADD / RSL-ADD is currently an OP3 workflow and G1 ADD training is disabled.
- Clarified that active RunPod workflows are PhysX-only and Newton scripts are disabled.
- Added hard sparse-dataset metadata requirements for embodiment and per-sequence FPS.
- Added actor-observable sparse orientation features and local root-velocity commands.
- Changed G1 preprocessing to build AMASS directly with the G1 embodiment profile instead of rescaling OP3 sparse AMASS.

## 0.1.0

- Initial scaffold for OP3 teleoperation locomotion in Isaac Lab.
- Retargeted the integration to Isaac Lab 2.3.2 with Isaac Sim/PhysX only.
- Reworked the custom RSL-ADD path for the legacy RSL-RL 3.0.1 API bundled with Isaac Lab 2.3.2.
- Added open dataset downloader and preprocessing hooks.
