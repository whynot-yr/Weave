"""Single-critic WEAVE MuonPPO with Getup terminal bootstrap; legacy class name retained."""
import torch
from tensordict import TensorDict
from .algorithm import MuonPPO


class MultiCriticMuonPPO(MuonPPO):
    def __init__(self, policy, **kwargs):
        super().__init__(policy, **kwargs)
        if self.rnd or self.symmetry or policy.is_recurrent:
            raise ValueError('Getup V1 supports feedforward PPO without RND/symmetry')

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.to(self.device).clone()
        self.transition.dones = dones
        mask = extras['bootstrap_timeouts'].to(self.device).bool()
        if mask.any():
            terminal = TensorDict({'robot_proprio': extras['terminal_proprio'].to(self.device)},
                                  batch_size=[dones.shape[0]])
            with torch.no_grad():
                value = self.policy.evaluate(terminal)
            self.transition.rewards += self.gamma * value.squeeze(-1) * mask
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
