from isaaclab.utils import configclass
from g1_hoi_learning.tasks.hoi.agents.rsl_rl_ppo_cfg import PPORunnerCfg, MuonPpoAlgorithmCfg, SimBaActorCriticCfg


@configclass
class MultiCriticAlgorithmCfg(MuonPpoAlgorithmCfg):
    class_name: str = 'MultiCriticMuonPPO'
    normalize_advantage_per_mini_batch: bool = False


@configclass
class GetupRunnerCfg(PPORunnerCfg):
    experiment_name = 'g1_inspire_getup'
    init_at_random_ep_len: bool = False
    obs_groups: dict = {'policy': ['robot_proprio'], 'critic': ['robot_proprio']}
    policy = SimBaActorCriticCfg(
        class_name='MultiCriticSimBaActorCritic', init_noise_std=.8,
        actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dim=2048, critic_hidden_dim=2048,
        actor_num_blocks=2, critic_num_blocks=2, expansion=1, latent_dim=256,
        encoder_hidden_dims={'robot_proprio': {'type': 'mlp', 'hidden_dims': [256]}},
    )
    algorithm = MultiCriticAlgorithmCfg(
        value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2,
        entropy_coef=.01, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=.001, schedule='adaptive', gamma=.99, lam=.95,
        desired_kl=.01, max_grad_norm=1., weight_decay=.01,
    )
