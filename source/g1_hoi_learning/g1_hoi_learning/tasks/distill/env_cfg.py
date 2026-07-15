from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from g1_hoi_learning.tasks.hoi.env_cfg import G1HoiLearningEnvCfg

from . import mdp
from .mdp.commands import GoalMotionCommandCfg


@configclass
class GoalCommandsCfg:
    """Goal-conditioned motion command (root + object-pose goal)."""

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
        joint_position_range=(-0.05, 0.05),
        object_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
        },
        gap=(50, 200),
    )


##
# Goal observation groups
##

@configclass
class GoalObservationsCfg:
    """Observation groups for distillation + RL::

        policy  (student) = goal_root + goal_object + object_state + robot_proprio
        critic            = ref_motion_body + ref_motion_object + object_state + robot_privileged
        teacher (frozen)  = ref_motion_body + ref_motion_object + object_state + robot_proprio
    """

    @configclass
    class GoalRootCfg(ObsGroup):
        """Sparse root (anchor) pose goal: residual current -> goal frame j, in the anchor frame."""

        goal_root_pos = ObsTerm(func=mdp.goal_root_pos_b, params={"command_name": "motion"})
        goal_root_rot = ObsTerm(func=mdp.goal_root_rot_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class GoalObjectCfg(ObsGroup):
        """Sparse object pose goal: residual current -> goal frame j, in the anchor frame."""

        goal_object_pos = ObsTerm(func=mdp.goal_object_pos_b, params={"command_name": "motion"})
        goal_object_rot = ObsTerm(func=mdp.goal_object_rot_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class RefMotionBodyCfg(ObsGroup):
        """Dense future reference of the robot bodies."""

        motion_future_joint_pos = ObsTerm(func=mdp.motion_future_joint_pos, params={"command_name": "motion"})
        motion_future_joint_vel = ObsTerm(func=mdp.motion_future_joint_vel, params={"command_name": "motion"})
        motion_future_anchor_pos_b = ObsTerm(func=mdp.motion_future_anchor_pos_b, params={"command_name": "motion"})
        motion_future_anchor_ori_b = ObsTerm(func=mdp.motion_future_anchor_ori_b, params={"command_name": "motion"})
        motion_future_body_pos_b = ObsTerm(func=mdp.motion_future_body_pos_b, params={"command_name": "motion"})
        motion_future_body_ori_b = ObsTerm(func=mdp.motion_future_body_ori_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class RefMotionObjectCfg(ObsGroup):
        """Dense future reference of the object, plus the reference contact labels."""

        motion_future_obj_pos_b = ObsTerm(func=mdp.motion_future_obj_pos_b, params={"command_name": "motion"})
        motion_future_obj_ori_b = ObsTerm(func=mdp.motion_future_obj_ori_b, params={"command_name": "motion"})
        motion_future_contact_label = ObsTerm(func=mdp.motion_future_contact_label, params={"command_name": "motion"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class ObjectStateCfg(ObsGroup):
        """Live object pose/shape, robot-object relational features, and live contact."""

        object_pos_b = ObsTerm(func=mdp.object_pos_b, params={"command_name": "motion"})
        object_rot_b = ObsTerm(func=mdp.object_rot_b, params={"command_name": "motion"})
        object_nearest_point_b = ObsTerm(func=mdp.object_nearest_point_b, params={"command_name": "motion"})
        object_point_cloud_b = ObsTerm(func=mdp.object_point_cloud_b, params={"command_name": "motion"})
        contact = ObsTerm(func=mdp.contact, params={"sensor_name": "contact_sensor"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class RobotProprioCfg(ObsGroup):
        """Deployable proprioception.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class RobotPrivilegedCfg(ObsGroup):
        """Ground-truth proprioception plus privileged information.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        actions = ObsTerm(func=mdp.last_action)

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    goal_root: GoalRootCfg = GoalRootCfg()
    goal_object: GoalObjectCfg = GoalObjectCfg()

    ref_motion_body: RefMotionBodyCfg = RefMotionBodyCfg()
    ref_motion_object: RefMotionObjectCfg = RefMotionObjectCfg()
    object_state: ObjectStateCfg = ObjectStateCfg()
    robot_proprio: RobotProprioCfg = RobotProprioCfg()
    robot_privileged: RobotPrivilegedCfg = RobotPrivilegedCfg()


##
# Environment configuration
##


@configclass
class G1HoiDistillEnvCfg(G1HoiLearningEnvCfg):
    """Goal-conditioned distillation + RL env."""

    commands: GoalCommandsCfg = GoalCommandsCfg()
    observations: GoalObservationsCfg = GoalObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
