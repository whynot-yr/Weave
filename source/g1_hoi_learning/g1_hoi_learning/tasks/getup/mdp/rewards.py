"""HoST Ground formulas, adapted to WEAVE joint/body names and 41 drivers."""
import math
import torch
from isaaclab.utils.math import quat_apply
from .state import get_state

# Group names are encoded in term names. Weights are internal, not advantage weights.
WEIGHTS = {
    'task_getup': 1.0,
    'style_waist': -10., 'style_hip_yaw': -10., 'style_hip_roll': -10.,
    'style_knee': -.25, 'style_shoulder_roll': -2.5,
    'style_left_foot': 2.5, 'style_right_foot': 2.5, 'style_feet_distance': -10.,
    'style_stumble': 0., 'style_shank': 10., 'style_ang_vel': 1.,
    'regu_acc': -2.5e-7, 'regu_action_rate': -.01, 'regu_smoothness': -.01,
    'regu_torque': -2.5e-6, 'regu_power': -2.5e-5, 'regu_vel': -1e-3,
    'regu_tracking': -.00025, 'regu_pos_limits': -100., 'regu_vel_limits': -1.,
    'post_ang_vel': 10., 'post_lin_vel': 10., 'post_orientation': 10.,
    'post_height': 10., 'post_upper': 10., 'post_feet_parallel': 2.5,
}


def tolerance_lower(x, lower, margin=1., value_at_margin=.1):
    distance = (lower-x).clamp_min(0)/margin
    return torch.exp(math.log(value_at_margin)*distance.square())


def compute_terms(env):
    s = get_state(env)
    d, cfg = s.robot.data, env.cfg.pose
    q, v, drive = d.joint_pos, d.joint_vel, s.drive
    p = d.body_link_pos_w
    origin = env.scene.env_origins
    root = d.root_pos_w-origin
    height = root[:, 2]
    post = height > cfg.post_height_gate
    style = height > cfg.style_height_gate
    feet = p[:, [s.body_ids['left_ankle_roll_link'], s.body_ids['right_ankle_roll_link']]]-origin[:, None]
    knees = p[:, [s.body_ids['left_knee_link'], s.body_ids['right_knee_link']]]-origin[:, None]
    torso = s.body_ids['torso_link']
    offset = torch.tensor(cfg.head_offset, device=env.device).expand(env.num_envs, -1)
    head = p[:, torso]+quat_apply(d.body_link_quat_w[:, torso], offset)-origin
    head_reward = tolerance_lower(head[:, 2]-feet[:, :, 2].mean(-1), cfg.head_target, cfg.head_margin)
    upright = tolerance_lower(-d.projected_gravity_b[:, 2], .99, 1., .05)

    def pair(part):
        return q[:, [s.joint_ids[f'left_{part}_joint'], s.joint_ids[f'right_{part}_joint']]]

    def hip_deviation(part):
        x = pair(part).abs()
        return ((x.amax(-1)>1.4) | (x.amin(-1)>.9)).float()

    knee = pair('knee')
    shoulder = pair('shoulder_roll')
    displacement = (root[:, None, :2]-feet[:, :, :2]).square().sum(-1).clamp_min(.3)
    foot_reward = torch.exp(-2*displacement)*(feet[:, :, 2]<.3)*post[:, None]
    shank = knees-feet
    vertical = (shank[:, :, 2]/shank.norm(dim=-1).clamp_min(1e-8)).mean(-1)
    angular = torch.exp(-2*d.root_ang_vel_b[:, :2].square().sum(-1))
    action = env.action_manager.action
    vel = v[:, drive]
    torque = d.applied_torque[:, drive]
    limits = d.soft_joint_pos_limits[:, drive]
    target = s.action.processed_actions
    terms = {
        'task_getup': head_reward*upright,
        'style_waist': (q[:, s.joint_ids['waist_yaw_joint']].abs()>1.4).float(),
        'style_hip_yaw': hip_deviation('hip_yaw'), 'style_hip_roll': hip_deviation('hip_roll'),
        'style_knee': ((knee.abs().amax(-1)>2.85)|(knee.amin(-1)<-.06)).float(),
        'style_shoulder_roll': ((shoulder[:, 0]<-.02)|(shoulder[:, 1]>.02)).float(),
        'style_left_foot': foot_reward[:, 0], 'style_right_foot': foot_reward[:, 1],
        'style_feet_distance': ((feet[:, 0]-feet[:, 1]).norm(dim=-1)>.9).float(),
        'style_stumble': torch.zeros_like(height),  # disabled in Ground V1
        'style_shank': tolerance_lower(vertical, .8)*style,
        'style_ang_vel': angular*style,
        'regu_acc': ((s.previous_velocity-vel)/env.step_dt).square().sum(-1),
        'regu_action_rate': (action-s.previous_action).square().sum(-1),
        'regu_smoothness': (action-2*s.previous_action+s.previous_previous_action).square().sum(-1),
        'regu_torque': torque.square().sum(-1), 'regu_power': (torque.abs()*vel.abs()).sum(-1),
        'regu_vel': vel.square().sum(-1), 'regu_tracking': (target-q[:, drive]).square().sum(-1),
        'regu_pos_limits': ((limits[..., 0]-q[:, drive]).clamp_min(0)+(q[:, drive]-limits[..., 1]).clamp_min(0)).sum(-1),
        'regu_vel_limits': (vel.abs()-.9*d.joint_vel_limits[:, drive]).clamp(0, 1).sum(-1),
        'post_ang_vel': angular*post,
        'post_lin_vel': torch.exp(-5*d.root_lin_vel_b[:, :2].square().sum(-1))*post,
        'post_orientation': torch.exp(-5*d.projected_gravity_b[:, :2].square().sum(-1))*post,
        'post_height': torch.exp(-20*(height-cfg.standing_height_target).abs())*post,
        'post_upper': torch.exp(-.1*(q[:, s.upper]-d.default_joint_pos[:, s.upper]).square().sum(-1))*post,
        'post_feet_parallel': torch.exp(-2*(10*(feet[:, 0, 2]-feet[:, 1, 2])).abs().clamp_min(.2))*post,
    }
    return terms


def reward_term(env, name):
    return get_state(env).terms[name]
