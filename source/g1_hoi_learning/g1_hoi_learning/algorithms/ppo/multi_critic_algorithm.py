"""WEAVE MuonPPO with four reward streams; no changes to installed RSL-RL."""
import torch
from tensordict import TensorDict
from .algorithm import MuonPPO
from .multi_critic_storage import MultiCriticRolloutStorage


class MultiCriticMuonPPO(MuonPPO):
    def __init__(self, policy, reward_group_weights=(2.5, 1.0, 0.1, 1.0), **kwargs):
        if kwargs.get('normalize_advantage_per_mini_batch', False):
            raise ValueError('HoST advantages must not be normalized a second time')
        super().__init__(policy, **kwargs)
        if self.rnd or self.symmetry or policy.is_recurrent:
            raise ValueError('Getup V1 supports feedforward PPO without RND/symmetry')
        self.reward_group_weights = reward_group_weights

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = MultiCriticRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device,
            reward_group_weights=self.reward_group_weights,
        )

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)
        self.transition.rewards = extras['reward_groups'].to(self.device).clone()
        self.transition.dones = dones
        mask = extras['bootstrap_timeouts'].to(self.device).bool()
        if mask.any():
            terminal = TensorDict({'robot_proprio': extras['terminal_proprio'].to(self.device)},
                                  batch_size=[dones.shape[0]])
            with torch.no_grad():
                value = self.policy.evaluate(terminal)
            self.transition.rewards += self.gamma * value * mask[:, None]
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        # Parent averages clipped value losses over batch and critic axes.
        metrics = {}
        with torch.no_grad():
            for i, name in enumerate(('task', 'style', 'regu', 'post')):
                value, target = self.storage.values[..., i], self.storage.returns[..., i]
                metrics[f'value_mse/{name}'] = (value-target).square().mean().item()
                metrics[f'explained_variance/{name}'] = (1-(target-value).var()/(target.var()+1e-8)).item()
        metrics.update(super().update())
        return metrics
