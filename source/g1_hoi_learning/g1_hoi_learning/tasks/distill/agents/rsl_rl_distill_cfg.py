"""RSL-RL cfg for the distillation+RL stage — SimBa actor-critic + frozen teacher + MuonPPODistill.
"""

from typing import Any

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ActorCriticTeacherCfg(RslRlPpoActorCriticCfg):
    class_name: str = "SimBaActorCriticTeacher"

    actor_hidden_dims: list[int] = []
    critic_hidden_dims: list[int] = []
    activation: str = "relu"

    actor_hidden_dim: int = 1024
    critic_hidden_dim: int = 1024
    actor_num_blocks: int = 1
    critic_num_blocks: int = 2
    expansion: int = 4
    teacher_hidden_dim: int = 1024
    teacher_num_blocks: int = 1

    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    teacher_obs_normalization: bool = True
    init_noise_std: float = 0.5
    noise_std_type: str = "scalar"

    latent_dim: int = 128
    encoder_hidden_dims: dict[str, Any] = {
        "ref_motion_body": {"type": "mlp", "hidden_dims": [512, 256]},
        "ref_motion_object": {"type": "mlp", "hidden_dims": [256]},
        "object_state": {"type": "mlp", "hidden_dims": [256]},
        "robot_proprio": {"type": "mlp", "hidden_dims": [256]},
        "robot_privileged": {"type": "mlp", "hidden_dims": [256]},
        "goal_root": {"type": "mlp", "hidden_dims": [256]},
        "goal_object": {"type": "mlp", "hidden_dims": [256]},
    }


@configclass
class MuonPPODistillCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "MuonPPODistill"
    weight_decay: float = 0.01
    bc_coef: float = 0.3
    bc_coef_min: float = 0.05        
    bc_coef_anneal_iters: int = 6000

    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 5.0e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class DistillRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "OnPolicyRunner"
    experiment_name: str = "g1_inspire_hoi_distill"
    num_steps_per_env: int = 32
    max_iterations: int = 100000
    save_interval: int = 200
    
    obs_groups: dict = {
        "policy": ["goal_root", "goal_object", "object_state", "robot_proprio"],
        "critic": ["ref_motion_body", "ref_motion_object", "object_state", "robot_privileged"],
        "teacher": ["ref_motion_body", "ref_motion_object", "object_state", "robot_proprio"],
    }
    policy: ActorCriticTeacherCfg = ActorCriticTeacherCfg()
    algorithm: MuonPPODistillCfg = MuonPPODistillCfg()
