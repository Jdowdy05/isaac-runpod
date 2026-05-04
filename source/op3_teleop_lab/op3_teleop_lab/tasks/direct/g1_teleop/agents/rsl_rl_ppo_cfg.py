from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1TeleopPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stock RSL-RL PPO config for G1 matched to the explicit OmniH2O PPO settings.

    Notes:
    - The public OmniH2O `ppo_teleop.yaml` explicitly specifies the 3-layer MLP,
      PPO optimizer hyperparameters, and `policy_class_name: ActorCritic`.
    - That same config also includes `rnn_type: lstm`, but without switching the
      policy class to a recurrent actor-critic or specifying recurrent hidden
      dimensions/layer count in the public training command. We therefore mirror
      the unambiguous published PPO settings here and keep the policy feedforward.
    """

    seed = 1
    num_steps_per_env = 24
    max_iterations = 10_000_000
    save_interval = 500
    experiment_name = "g1_teleop_rsl_rl"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 100.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.2,
    )
