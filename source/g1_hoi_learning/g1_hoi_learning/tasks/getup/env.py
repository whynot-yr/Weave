import warnings
from isaaclab.envs import ManagerBasedRLEnv
from .reward_manager import GetupRewardManager
from .mdp.state import get_state


class GetupEnv(ManagerBasedRLEnv):
    def load_managers(self):
        super().load_managers()
        self.reward_manager = GetupRewardManager(self.cfg.rewards, self)
        if not self.cfg.pose.calibration_verified:
            warnings.warn('Getup V1 uses provisional head/height calibration. Validate poses before long training.')

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        state = get_state(self)
        # Attribute terminal transitions to the OLD seed before reset changes it.
        completed = (self.episode_length_buf[env_ids] > 0) & (
            self.reset_terminated[env_ids] | self.reset_time_outs[env_ids])
        old_seed = state.episode_seed_id[env_ids]
        for seed_id in range(2):
            ids = env_ids[completed & (old_seed == seed_id)]
            if len(ids) == 0:
                continue
            state.seed_stats[seed_id, 1] += len(ids)
            state.seed_stats[seed_id, 2] += state.success[ids].sum()
            state.seed_stats[seed_id, 3] += (state.success_steps[ids] >= state.required_steps).sum()
            state.seed_stats[seed_id, 4] += self.episode_length_buf[ids].sum()
            state.seed_stats[seed_id, 5:] += self.reward_manager.group_sums[ids].sum(dim=0)
        success = state.success[env_ids].float().mean().item()
        final_standing = (state.success_steps[env_ids] >= state.required_steps).float().mean().item()
        super()._reset_idx(env_ids)
        self.extras['log']['Success/episode'] = success
        self.extras['log']['Success/final_hold'] = final_standing

    def pop_seed_statistics(self):
        """Consume additive counts/sums. No distributed collectives during asynchronous resets."""
        state = get_state(self)
        stats = state.seed_stats.clone()
        state.seed_stats.zero_()
        return stats
