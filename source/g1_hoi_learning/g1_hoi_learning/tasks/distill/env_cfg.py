# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger distillation env cfg — subclasses the ``hoi`` tracking env.

Two observation groups:
  - ``policy``  : the STUDENT observes sparse object-pose goal and
                  proprio + live object + contact; NO dense reference.
  - ``teacher`` : the TEACHER observes dense future reference. 
                  Consumed by the frozen teacher.

Rewards/terminations are inherited unchanged from the tracker.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from g1_hoi_learning.tasks.hoi.env_cfg import G1HoiLearningEnvCfg
from g1_hoi_learning.tasks.hoi.env_cfg import ObservationsCfg as ExpertPolicyCfg

from . import mdp
from .mdp.commands import GoalMotionCommandCfg


@configclass
class GoalCommandsCfg:
    """Goal-conditioned motion command (object-pose goal)."""

    motion = GoalMotionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        rsi=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        joint_position_range=(-0.1, 0.1),
        gap=(50, 200),
    )


@configclass
class GoalPolicyCfg(ObsGroup):
    """Deployable student obs: sparse object-pose goal + proprio + live object + contact (no dense reference)."""

    # sparse goal
    goal_object_pos = ObsTerm(func=mdp.goal_object_pos_b, params={"command_name": "motion"})
    goal_object_rot = ObsTerm(func=mdp.goal_object_rot_b, params={"command_name": "motion"})
    # contact
    contact = ObsTerm(func=mdp.contact, params={"sensor_name": "contact_sensor"})
    # robot state
    body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
    body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    # object state
    object_pos_b = ObsTerm(func=mdp.object_pos_b, params={"command_name": "motion"})
    object_rot_b = ObsTerm(func=mdp.object_rot_b, params={"command_name": "motion"})
    object_nearest_point_b = ObsTerm(func=mdp.object_nearest_point_b, params={"command_name": "motion"})
    object_point_cloud_b = ObsTerm(func=mdp.object_point_cloud_b, params={"command_name": "motion"})
    # proprioception
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class GoalObservationsCfg:
    """policy = student (deployable); teacher = tracker's exact PolicyCfg (dense reference)."""

    policy: GoalPolicyCfg = GoalPolicyCfg()
    teacher: ExpertPolicyCfg.PolicyCfg = ExpertPolicyCfg.PolicyCfg()

    def __post_init__(self):
        # clean teacher labels: no obs corruption on the frozen teacher's obs
        self.teacher.enable_corruption = False


##
# Environment configuration
##


@configclass
class G1HoiDistillEnvCfg(G1HoiLearningEnvCfg):
    """Goal-conditioned DAgger distillation env."""

    commands: GoalCommandsCfg = GoalCommandsCfg()
    observations: GoalObservationsCfg = GoalObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
