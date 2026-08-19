from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .mdp.commands import MotionCommandCfg

##
# Pre-defined configs
##
from g1_hoi_learning.robots.g1_inspire import G1_INSPIRE_CFG  # isort:skip
from g1_hoi_learning.assets import GROUND_PLANE_USD_PATH

##
# Scene definition
##


@configclass
class G1HoiLearningSceneCfg(InteractiveSceneCfg):
    """Configuration for a G1 HOI learning scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            usd_path=GROUND_PLANE_USD_PATH,
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    robot: ArticulationCfg = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object: RigidObjectCfg = MISSING   # set in G1HoiLearningEnvCfg.__post_init__ from motion_file's object_name

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        history_length=0,
        track_air_time=False,
        filter_prim_paths_expr=[
            # robot.body_names order
            "{ENV_REGEX_NS}/Robot/pelvis",
            "{ENV_REGEX_NS}/Robot/left_hip_pitch_link", "{ENV_REGEX_NS}/Robot/right_hip_pitch_link",
            "{ENV_REGEX_NS}/Robot/waist_yaw_link",
            "{ENV_REGEX_NS}/Robot/left_hip_roll_link", "{ENV_REGEX_NS}/Robot/right_hip_roll_link",
            "{ENV_REGEX_NS}/Robot/waist_roll_link",
            "{ENV_REGEX_NS}/Robot/left_hip_yaw_link", "{ENV_REGEX_NS}/Robot/right_hip_yaw_link",
            "{ENV_REGEX_NS}/Robot/torso_link",
            "{ENV_REGEX_NS}/Robot/left_knee_link", "{ENV_REGEX_NS}/Robot/right_knee_link",
            "{ENV_REGEX_NS}/Robot/left_shoulder_pitch_link", "{ENV_REGEX_NS}/Robot/right_shoulder_pitch_link",
            "{ENV_REGEX_NS}/Robot/left_ankle_pitch_link", "{ENV_REGEX_NS}/Robot/right_ankle_pitch_link",
            "{ENV_REGEX_NS}/Robot/left_shoulder_roll_link", "{ENV_REGEX_NS}/Robot/right_shoulder_roll_link",
            "{ENV_REGEX_NS}/Robot/left_ankle_roll_link", "{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            "{ENV_REGEX_NS}/Robot/left_shoulder_yaw_link", "{ENV_REGEX_NS}/Robot/right_shoulder_yaw_link",
            "{ENV_REGEX_NS}/Robot/left_elbow_link", "{ENV_REGEX_NS}/Robot/right_elbow_link",
            "{ENV_REGEX_NS}/Robot/left_wrist_roll_link", "{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
            "{ENV_REGEX_NS}/Robot/left_wrist_pitch_link", "{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
            "{ENV_REGEX_NS}/Robot/left_wrist_yaw_link", "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
            "{ENV_REGEX_NS}/Robot/L_index_proximal", "{ENV_REGEX_NS}/Robot/L_middle_proximal",
            "{ENV_REGEX_NS}/Robot/L_pinky_proximal", "{ENV_REGEX_NS}/Robot/L_ring_proximal",
            "{ENV_REGEX_NS}/Robot/L_thumb_proximal_base",
            "{ENV_REGEX_NS}/Robot/R_index_proximal", "{ENV_REGEX_NS}/Robot/R_middle_proximal",
            "{ENV_REGEX_NS}/Robot/R_pinky_proximal", "{ENV_REGEX_NS}/Robot/R_ring_proximal",
            "{ENV_REGEX_NS}/Robot/R_thumb_proximal_base",
            "{ENV_REGEX_NS}/Robot/L_index_intermediate", "{ENV_REGEX_NS}/Robot/L_middle_intermediate",
            "{ENV_REGEX_NS}/Robot/L_pinky_intermediate", "{ENV_REGEX_NS}/Robot/L_ring_intermediate",
            "{ENV_REGEX_NS}/Robot/L_thumb_proximal",
            "{ENV_REGEX_NS}/Robot/R_index_intermediate", "{ENV_REGEX_NS}/Robot/R_middle_intermediate",
            "{ENV_REGEX_NS}/Robot/R_pinky_intermediate", "{ENV_REGEX_NS}/Robot/R_ring_intermediate",
            "{ENV_REGEX_NS}/Robot/R_thumb_proximal",
            "{ENV_REGEX_NS}/Robot/L_thumb_intermediate", "{ENV_REGEX_NS}/Robot/R_thumb_intermediate",
            "{ENV_REGEX_NS}/Robot/L_thumb_distal", "{ENV_REGEX_NS}/Robot/R_thumb_distal",
        ],
    )

    feet_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=False,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = MotionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        rsi=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.MimicJointPositionActionCfg(
        asset_name="robot",
        # Exclude passive Inspire hand joints (intermediate/distal) — driven via the mimic table.
        joint_names=["^(?!.*(intermediate|distal)).*$"],
        mimic={
            "L_thumb_intermediate_joint":  ("L_thumb_proximal_pitch_joint", 1.6, 0.0),
            "L_thumb_distal_joint":        ("L_thumb_proximal_pitch_joint", 2.4, 0.0),
            "L_index_intermediate_joint":  ("L_index_proximal_joint",       1.0, 0.0),
            "L_middle_intermediate_joint": ("L_middle_proximal_joint",      1.0, 0.0),
            "L_ring_intermediate_joint":   ("L_ring_proximal_joint",        1.0, 0.0),
            "L_pinky_intermediate_joint":  ("L_pinky_proximal_joint",       1.0, 0.0),
            "R_thumb_intermediate_joint":  ("R_thumb_proximal_pitch_joint", 1.6, 0.0),
            "R_thumb_distal_joint":        ("R_thumb_proximal_pitch_joint", 2.4, 0.0),
            "R_index_intermediate_joint":  ("R_index_proximal_joint",       1.0, 0.0),
            "R_middle_intermediate_joint": ("R_middle_proximal_joint",      1.0, 0.0),
            "R_ring_intermediate_joint":   ("R_ring_proximal_joint",        1.0, 0.0),
            "R_pinky_intermediate_joint":  ("R_pinky_proximal_joint",       1.0, 0.0),
        },
    )


@configclass
class ObservationsCfg:
    """Observation groups::

        actor  = ref_motion_body + ref_motion_object + object_state + robot_proprio
        critic = ref_motion_body + ref_motion_object + object_state + robot_privileged
    """

    @configclass
    class RefMotionBodyCfg(ObsGroup):
        """Dense future reference of the robot bodies."""

        motion_future_joint_pos = ObsTerm(func=mdp.motion_future_joint_pos, params={"command_name": "motion"})
        motion_future_anchor_pos_b = ObsTerm(func=mdp.motion_future_anchor_pos_b, params={"command_name": "motion"})
        motion_future_anchor_ori_b = ObsTerm(func=mdp.motion_future_anchor_ori_b, params={"command_name": "motion"})

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
        object_nearest_point_b = ObsTerm(
            func=mdp.object_nearest_point_b,
            params={
                "command_name": "motion",
                "body_names": [".*_thumb_distal", ".*_(index|middle|ring|pinky)_intermediate"],
            },
        )
        object_bps_sdf_b = ObsTerm(func=mdp.ObjectBpsSdf, params={"command_name": "motion"})
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

    ref_motion_body: RefMotionBodyCfg = RefMotionBodyCfg()
    ref_motion_object: RefMotionObjectCfg = RefMotionObjectCfg()
    object_state: ObjectStateCfg = ObjectStateCfg()
    robot_proprio: RobotProprioCfg = RobotProprioCfg()
    robot_privileged: RobotPrivilegedCfg = RobotPrivilegedCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    randomize_finger_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thumb|index|middle|ring|pinky)_.*_joint"),
            "stiffness_distribution_params": (0.5, 2.0),   # ×default 5.0
            "damping_distribution_params": (0.5, 2.0),     # ×default 0.5
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_finger_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thumb|index|middle|ring|pinky)_.*_joint"),
            "armature_distribution_params": (0.5, 1.5),    # ×default 0.01
            "operation": "scale",
            "distribution": "uniform",
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_anchor_pos = RewTerm(
        func=mdp.motion_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_anchor_ori = RewTerm(
        func=mdp.motion_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    # object tracking
    object_pos = RewTerm(
        func=mdp.object_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    object_ori = RewTerm(
        func=mdp.object_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    # reference-free grasp shape: thumb opposing the other fingers across the object surface
    hand_opposition = RewTerm(
        func=mdp.hand_opposition_reward,
        weight=2.0,
        params={
            "command_name": "motion",
            "thumb_body_name": "thumb_distal",
            "finger_body_names": [
                "index_intermediate", "middle_intermediate", "ring_intermediate", "pinky_intermediate",
            ],
        },
    )
    # contact
    contact = RewTerm(
        func=mdp.contact_reward,
        weight=2.0,
        params={
            "command_name": "motion",
            "sensor_name": "contact_sensor",
            "hand_body_names": [".*_thumb_intermediate", ".*_thumb_distal", ".*_index_intermediate", ".*_middle_intermediate", ".*_ring_intermediate", ".*_pinky_intermediate"],
            "saturate_force": 5.0,
        },
    )
    # regularization
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*(thumb|index|middle|ring|pinky)).*$"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    object_pos = DoneTerm(
        func=mdp.bad_object_pos,
        params={"command_name": "motion", "threshold": 0.25},
    )
    object_ori = DoneTerm(
        func=mdp.bad_object_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.3},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )
    bad_contact = DoneTerm(
        func=mdp.bad_contact,
        params={
            "command_name": "motion",
            "sensor_name": "contact_sensor",
            "hand_body_names": [".*_thumb_intermediate", ".*_thumb_distal", ".*_index_intermediate", ".*_middle_intermediate", ".*_ring_intermediate", ".*_pinky_intermediate"],
            "max_lost_frames": 10,
        },
    )


##
# Environment configuration
##


@configclass
class G1HoiLearningEnvCfg(ManagerBasedRLEnvCfg):
    """G1 HOI learning environment config."""

    scene: G1HoiLearningSceneCfg = G1HoiLearningSceneCfg(num_envs=4096, env_spacing=4.0)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (-3.0, -3.0, 2.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 16 * 2**16
