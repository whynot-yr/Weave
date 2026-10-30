def joint_velocity_out(env):
    return env.scene['robot'].data.joint_vel.abs().amax(-1) > 300.0


def base_velocity_out(env):
    return env.scene['robot'].data.root_lin_vel_w.norm(dim=-1) > 20.0
