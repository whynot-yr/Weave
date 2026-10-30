import torch
from isaaclab.managers import RewardManager
from .mdp.state import get_state
from .mdp.rewards import compute_terms

GROUPS = ('task', 'style', 'regu', 'post')


class GetupRewardManager(RewardManager):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.group_sums = torch.zeros(env.num_envs, 4, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        logs = {f'Getup_Return/{name}': self.group_sums[ids, k].mean().item()
                for k, name in enumerate(GROUPS)}
        self.group_sums[ids] = 0
        logs.update(super().reset(env_ids))
        return logs

    def compute(self, dt):
        s = get_state(self._env)
        s.begin_step()
        s.terms = compute_terms(self._env)
        groups = torch.zeros(self.num_envs, 4, device=self.device)
        for i, (name, cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            k = GROUPS.index(name.split('_')[0])
            value = s.terms[name]*cfg.weight*(1.0 if k == 0 else dt)
            groups[:, k] += value
            self._episode_sums[name] += value
            self._step_reward[:, i] = value
        self._reward_buf[:] = (groups*torch.tensor(self._env.cfg.advantage_weights, device=self.device)).sum(-1)
        self.group_sums += groups
        self._env.extras['reward_groups'] = groups
        self._env.extras['bootstrap_timeouts'] = self._env.reset_time_outs & ~self._env.reset_terminated
        # Capture the completed transition before any automatic reset.
        self._env.extras['terminal_proprio'] = s.observation()
        s.end_step()
        return self._reward_buf
