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

