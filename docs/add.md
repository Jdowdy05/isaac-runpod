# ADD Notes

This repository includes a standalone implementation of the paper:

- [Physics-Based Motion Imitation with Adversarial Differential Discriminators](https://arxiv.org/abs/2505.04961)

## What Is Implemented Here

- PPO actor-critic training
- a discriminator over normalized differential tracking vectors
- zero-vector positive samples for the discriminator
- discriminator replay buffer
- differential normalizer based on mean absolute value

## What It Is Not

- not a teacher-student distillation pipeline
- not pure supervised MSE fitting to the dataset

The current OP3 integration uses the sparse teleoperation target as the reference and builds the differential vector from:

- tracked body position differences
- tracked body orientation differences
- target linear velocity differences

