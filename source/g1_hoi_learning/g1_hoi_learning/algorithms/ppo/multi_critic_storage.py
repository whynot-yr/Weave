"""RSL-RL storage with per-group returns and HoST advantage aggregation."""
import torch
from rsl_rl.storage import RolloutStorage


class MultiCriticRolloutStorage(RolloutStorage):
    def __init__(self, *args, reward_group_weights, **kwargs):
        super().__init__(*args, **kwargs)
        if self.training_type != 'rl':
            raise ValueError('Multi-critic storage only supports RL')
        shape = (self.num_transitions_per_env, self.num_envs, 4)
        for name in ('rewards', 'values', 'returns', 'group_advantages'):
            setattr(self, name, torch.zeros(shape, device=self.device))
        self.group_weights = torch.as_tensor(reward_group_weights, device=self.device)
        if self.group_weights.shape != (4,):
            raise ValueError('Expected four group weights: task/style/regu/post')

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError('Rollout buffer full')
        if transition.rewards.shape != (self.num_envs, 4):
            raise ValueError(f'Expected [N,4] rewards, got {transition.rewards.shape}')
        self.observations[self.step].copy_(transition.observations)
        for field, source in [('actions', 'actions'), ('rewards', 'rewards'), ('values', 'values'),
                              ('mu', 'action_mean'), ('sigma', 'action_sigma')]:
            getattr(self, field)[self.step].copy_(getattr(transition, source))
        self.dones[self.step].copy_(transition.dones.reshape(-1, 1))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.reshape(-1, 1))
        self.step += 1

    def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
        if self.step != self.num_transitions_per_env:
            raise ValueError('Compute returns only for a complete rollout')
        advantage = torch.zeros_like(last_values)
        for t in reversed(range(self.num_transitions_per_env)):
            next_value = last_values if t == self.num_transitions_per_env - 1 else self.values[t+1]
            live = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * live * next_value - self.values[t]
            advantage = delta + gamma * lam * live * advantage
            self.returns[t] = advantage + self.values[t]
        raw = self.returns - self.values
        # HoST: normalize each reward stream over the rollout, then weight/sum.
        std = raw.std(dim=(0, 1), correction=1 if raw.shape[0]*raw.shape[1] > 1 else 0)
        self.group_advantages.copy_((raw - raw.mean(dim=(0, 1))) / (std + 1e-8))
        self.advantages = (self.group_advantages * self.group_weights).sum(-1, keepdim=True)
