"""RSL-RL cfg for the distillation+RL stage — SimBa actor-critic + frozen teacher + MuonPPODistill.
"""

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

    actor_hidden_dim: int = 2048
    critic_hidden_dim: int = 2048
    actor_num_blocks: int = 2
    critic_num_blocks: int = 2
    expansion: int = 1
    teacher_hidden_dim: int = 2048
    teacher_num_blocks: int = 2

    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    teacher_obs_normalization: bool = True
    init_noise_std: float = 0.5
    noise_std_type: str = "scalar"


@configclass
class MuonPPODistillCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "MuonPPODistill"
    weight_decay: float = 0.01
    bc_coef: float = 1.0            
    bc_coef_decay: float = 1.0      
    
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 5.0e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
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
    
    obs_groups: dict = {"policy": ["policy"], "critic": ["teacher"], "teacher": ["teacher"]}
    policy: ActorCriticTeacherCfg = ActorCriticTeacherCfg()
    algorithm: MuonPPODistillCfg = MuonPPODistillCfg()
