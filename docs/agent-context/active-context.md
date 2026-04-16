# Active Context

Last updated: 2026-04-16

## Project Snapshot

- Repository: `CustomVLAProject`
- Focus: Establish reusable agent skills and a persistent research notebook for coding, robotics, reinforcement learning, imitation learning, paper digestion, and idea development.

## Current Priorities

- Use installed official skills for notebooks, PDFs, GitHub workflows, and Notion-backed research capture.
- Use custom local skills for robotics/RL/IL planning, paper comprehension, and idea-to-experiment conversion.
- Keep this notebook current when decisions or experiments produce durable context.
- Draft the IEEE paper in `IEEE___Full_Body_TeleOP3` around the current OP3 full-body teleoperation stack.
- Position the paper around the miniature-humanoid challenge: low torque, reduced morphology similarity to humans, and a harder embodiment transfer problem than in recent full-sized humanoid teleoperation work.
- Debug the OP3 Newton environment for possible ground-contact failure after playback showed the robot falling through the floor.
- Use the standalone diagnostic script `scripts/debug/test_newton_ground_contact.py` and the RunPod wrapper `scripts/runpod/test_newton_ground_contact.sh` to separate terrain/collision issues from policy or recording issues.
- Use PhysX as the default OP3 backend for training and playback until the Newton collision issue is resolved.

## Open Questions

- Which robotics stack, simulator, and benchmark tasks matter most for this repository?
- What are the immediate build targets: literature review, policy training, sim infrastructure, real robot integration, or data collection?
- What final experimental evidence will be reported in the paper: simulation-only results, real OP3 teleoperation, or both?
- Which figures should anchor the manuscript: system diagram, sparse command representation, dataset filtering pipeline, or qualitative rollout frames?

## Next Useful Actions

- Record important project decisions here as the repository direction becomes clearer.
- Add paper summaries to `paper-notes.md`.
- Add experiment outcomes to `experiment-log.md`.
- Add candidate projects and hypotheses to `idea-backlog.md`.
- Add final quantitative plots and tables once the OP3 teleoperation experiments are stable enough for paper reporting.
- Run the new Newton ground-contact diagnostic on RunPod and inspect whether `/World/ground` is valid, collision-enabled, and able to support the OP3 spawn configuration.
- Investigate the OP3 USD collision setup under Newton, since the standalone diagnostic confirmed the plane exists and OP3 still falls through it in a minimal no-training test.
