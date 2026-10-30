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
        state = get_state(self)
        success = state.success[env_ids].float().mean().item()
        final_standing = (state.success_steps[env_ids] >= state.required_steps).float().mean().item()
        super()._reset_idx(env_ids)
        self.extras['log']['Success/episode'] = success
        self.extras['log']['Success/final_hold'] = final_standing
