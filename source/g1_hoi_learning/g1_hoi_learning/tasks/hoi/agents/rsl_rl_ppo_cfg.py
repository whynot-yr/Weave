from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MuonPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm cfg routed through MuonPPO.
    """

    class_name: str = "MuonPPO"
    weight_decay: float = 0.01


@configclass
class SimBaActorCriticCfg(RslRlPpoActorCriticCfg):
    """Config for the SimBa actor-critic backbone.
    """

    class_name: str = "SimBaActorCritic"

    # Unused by SimBa
    actor_hidden_dims: list[int] = []
    critic_hidden_dims: list[int] = []
    activation: str = "relu"

    # SimBa-specific
    actor_hidden_dim: int = 2048
    critic_hidden_dim: int = 2048
    actor_num_blocks: int = 2
    critic_num_blocks: int = 2
    expansion: int = 1

    latent_dim: int = 256
    encoder_hidden_dims: dict[str, list[int]] = {
        "ref_motion_body": [1024, 512],
        "ref_motion_object": [512],
        "object_state": [512],
        "robot_proprio": [512],
        "robot_privileged": [512],
    }


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 100
    experiment_name = "g1_inspire_hoi"
    obs_groups: dict = {
        "policy": ["ref_motion_body", "ref_motion_object", "object_state", "robot_proprio"],
        "critic": ["ref_motion_body", "ref_motion_object", "object_state", "robot_privileged"],
    }
    policy = SimBaActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dim=2048,
        critic_hidden_dim=2048,
        actor_num_blocks=2,
        critic_num_blocks=2,
        expansion=1,
        latent_dim=256,
    )
    algorithm = MuonPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=5e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        weight_decay=0.01,
    )
