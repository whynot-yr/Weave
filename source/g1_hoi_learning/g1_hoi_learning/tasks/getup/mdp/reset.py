import torch
from .state import get_state


def reset_getup(env, env_ids):
    state = get_state(env)
    robot, cfg = state.robot, env.cfg.pose
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    if cfg.reset_mode == 'mixed':
        if not 0.0 <= cfg.prone_probability <= 1.0:
            raise ValueError('prone_probability must be in [0, 1]')
        seed_ids = (torch.rand(n, device=env.device) < cfg.prone_probability).long()
    elif cfg.reset_mode in state.seed_names:
        seed_ids = torch.full((n,), state.seed_names.index(cfg.reset_mode), device=env.device, dtype=torch.long)
    else:
        raise ValueError(f'Unknown getup reset_mode: {cfg.reset_mode}')
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] = state.seed_root_pos[seed_ids] + env.scene.env_origins[env_ids]
    root[:, 3:7] = state.seed_root_quat[seed_ids]
    root[:, 7:] = 0
    q = robot.data.default_joint_pos[env_ids].clone()
    q[:, state.drive] = state.seed_joint_pos[seed_ids]
    limits = robot.data.joint_pos_limits[env_ids].clone()
    mimic = state.action.cfg.mimic
    for name, (driver, multiplier, offset) in mimic.items():
        j, d = state.joint_ids[name], state.joint_ids[driver]
        bounds = (limits[:, j]-offset)/multiplier
        limits[:, d, 0] = torch.maximum(limits[:, d, 0], bounds.min(-1).values)
        limits[:, d, 1] = torch.minimum(limits[:, d, 1], bounds.max(-1).values)
    if cfg.randomize_joints:
        lo, hi = cfg.joint_scale_range
        ol, oh = cfg.joint_offset_range
        q[:, state.drive] = q[:, state.drive]*(lo+(hi-lo)*torch.rand(n, 41, device=env.device)) \
            + ol+(oh-ol)*torch.rand(n, 41, device=env.device)
    q[:, state.drive] = q[:, state.drive].clamp(limits[:, state.drive, 0], limits[:, state.drive, 1])
    for name, (driver, multiplier, offset) in mimic.items():
        q[:, state.joint_ids[name]] = q[:, state.joint_ids[driver]]*multiplier+offset
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    robot.set_joint_position_target(q, env_ids=env_ids)
    state.reset(env_ids)
    state.episode_seed_id[env_ids] = seed_ids
    state.seed_stats[:, 0] += torch.bincount(seed_ids, minlength=2)
