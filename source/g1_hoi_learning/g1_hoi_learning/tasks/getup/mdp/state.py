"""Task history updated once per control step, with pre-reset terminal snapshots."""
import math
import torch


class GetupState:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene['robot']
        self.action = env.action_manager.get_term('joint_pos')
        self.drive = list(self.action._joint_ids)
        if self.robot.num_joints != 53 or len(self.drive) != 41:
            raise ValueError('Getup expects 53 state joints and 41 actions')
        self.history = torch.zeros(env.num_envs, 3, 154, device=env.device)
        self.pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.previous_velocity = torch.zeros(env.num_envs, 41, device=env.device)
        self.previous_action = torch.zeros_like(self.previous_velocity)
        self.previous_previous_action = torch.zeros_like(self.previous_velocity)
        self.success_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.success = torch.zeros_like(self.pending)
        self.required_steps = math.ceil(env.cfg.pose.success_hold_s/env.step_dt)
        self.last_step = -1
        self.terms = {}
        self.joint_ids = {name: i for i, name in enumerate(self.robot.joint_names)}
        self.body_ids = {name: i for i, name in enumerate(self.robot.body_names)}
        self.upper = [i for name, i in self.joint_ids.items() if name.startswith('waist_') or
                      any(part in name for part in ('shoulder_', 'elbow_', 'wrist_'))]
        if len(self.upper) != 17:
            raise ValueError(f'Expected 17 upper-body joints, got {len(self.upper)}')

    def frame(self):
        d = self.robot.data
        return torch.cat((d.root_ang_vel_b, d.projected_gravity_b,
                          d.joint_pos-d.default_joint_pos, d.joint_vel-d.default_joint_vel,
                          (d.root_pos_w[:, 2]-self.env.scene.env_origins[:, 2])[:, None],
                          self.env.action_manager.action), dim=-1)

    def observation(self):
        if self.pending.any():
            frame = self.frame()
            self.history[self.pending] = frame[self.pending, None, :].expand(-1, 3, -1)
            self.pending[:] = False
        return self.history.flatten(1).clone()

    def begin_step(self):
        if self.last_step == self.env.common_step_counter:
            raise RuntimeError('Getup state advanced twice in one control step')
        self.last_step = self.env.common_step_counter
        frame = self.frame()
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = frame
        self.history[self.pending] = frame[self.pending, None, :].expand(-1, 3, -1)
        self.pending[:] = False
        high = frame[:, 112] > self.env.cfg.pose.success_height  # 3+3+53+53
        self.success_steps.copy_(torch.where(high, self.success_steps+1, 0))
        self.success |= self.success_steps >= self.required_steps

    def end_step(self):
        self.previous_previous_action.copy_(self.previous_action)
        self.previous_action.copy_(self.env.action_manager.action)
        self.previous_velocity.copy_(self.robot.data.joint_vel[:, self.drive])

    def reset(self, ids):
        self.pending[ids] = True
        self.previous_velocity[ids] = 0
        self.previous_action[ids] = 0
        self.previous_previous_action[ids] = 0
        self.success_steps[ids] = 0
        self.success[ids] = False


def get_state(env):
    if not hasattr(env, 'getup_state'):
        env.getup_state = GetupState(env)
    return env.getup_state
