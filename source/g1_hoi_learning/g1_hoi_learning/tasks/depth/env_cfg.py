"""Depth-perception env."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass

from g1_hoi_learning.tasks.hoi.env_cfg import CommandsCfg, EventCfg, G1HoiLearningEnvCfg, G1HoiLearningSceneCfg, ObservationsCfg

from . import mdp
from .mdp.commands import DepthMotionCommandCfg
from .mdp.depth_camera import ArticulationRayCasterCameraCfg, ArticulationTargetCfg, RaycastTargetCfg


@configclass
class DepthSceneCfg(G1HoiLearningSceneCfg):
    """hoi scene + head-mounted raycast depth camera (D435i on torso)."""

    depth_cam = ArticulationRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        update_period=1.0 / 30.0,
        debug_vis=False,
        offset=ArticulationRayCasterCameraCfg.OffsetCfg(
            pos=(0.0576235, 0.01753, 0.42987),
            rot=(0.91496, 0.0, 0.40355, 0.0),
            convention="world",  # forward +X, up +Z
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=3.0,
        pattern_cfg=PinholeCameraPatternCfg(
            width=128,
            height=72,
            focal_length=1.0,  # focal=1 => aperture = 2*tan(FOV/2)
            horizontal_aperture=2 * math.tan(math.radians(86.0) / 2),  # FOV_h 86 deg
            vertical_aperture=2 * math.tan(math.radians(57.0) / 2),  # FOV_v 57 deg
        ),
        mesh_prim_paths=[
            RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Object", is_shared=False, track_mesh_transforms=True),
            ArticulationTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/(?!torso_link$).*_link", is_shared=True),
            ArticulationTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/[LR]_.*", is_shared=True),
            RaycastTargetCfg(prim_expr="/World/ground", is_shared=True, track_mesh_transforms=False),
        ],
    )


@configclass
class DepthObservationsCfg(ObservationsCfg):
    """Raycast depth observation group."""

    @configclass
    class NoisyRefMotionBodyCfg(ObsGroup):
        """Future robot reference expressed in the student's noisy odometry frame."""

        motion_future_joint_pos = ObsTerm(func=mdp.motion_future_joint_pos, params={"command_name": "motion"})
        motion_future_anchor_pos_b = ObsTerm(
            func=mdp.motion_future_anchor_pos_noisy_b, params={"command_name": "motion"}
        )
        motion_future_anchor_ori_b = ObsTerm(
            func=mdp.motion_future_anchor_ori_noisy_b, params={"command_name": "motion"}
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class NoisyRefMotionObjectCfg(ObsGroup):
        """Future object reference expressed in the student's noisy odometry frame."""

        motion_future_obj_pos_b = ObsTerm(
            func=mdp.motion_future_obj_pos_noisy_b, params={"command_name": "motion"}
        )
        motion_future_obj_ori_b = ObsTerm(
            func=mdp.motion_future_obj_ori_noisy_b, params={"command_name": "motion"}
        )
        motion_future_contact_label = ObsTerm(func=mdp.motion_future_contact_label, params={"command_name": "motion"})

        def __post_init__(self):
            self.concatenate_terms = True

    noisy_ref_motion_body: NoisyRefMotionBodyCfg = NoisyRefMotionBodyCfg()
    noisy_ref_motion_object: NoisyRefMotionObjectCfg = NoisyRefMotionObjectCfg()

    @configclass
    class DepthCfg(ObsGroup):
        depth = ObsTerm(
            func=mdp.object_depth_b,
            params={
                "sensor_name": "depth_cam",
                "max_dist": 3.0,
                "min_z": 0.25,
                "noise_k_range": (0.005, 0.015),
                "dropout_prob": 0.05,
                "debug_vis": False,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()


@configclass
class DepthEventsCfg(EventCfg):
    """hoi events + per-reset domain randomization of the depth camera's extrinsics / intrinsics."""

    randomize_actor_odometry_bias = EventTerm(
        func=mdp.randomize_actor_odometry_bias,
        mode="reset",
        params={
            "command_name": "motion",
            "odom_pos_range": (0.05, 0.05, 0.015),
            "odom_rot_range": (0.015, 0.015, 0.045),
        },
    )

    randomize_camera_extrinsics = EventTerm(
        func=mdp.randomize_camera_extrinsics,
        mode="reset",
        params={
            "sensor_name": "depth_cam",
            "pos_noise_std": (0.02, 0.02, 0.02),  # mount position (m)
            "rot_noise_std": (0.03, 0.04, 0.03),  # mount orientation rpy (rad)
        },
    )

    randomize_camera_intrinsics = EventTerm(
        func=mdp.randomize_camera_intrinsics,
        mode="reset",
        params={
            "sensor_name": "depth_cam",
            "focal_length_noise_std": 0.02,
            "aperture_noise_std": 0.05,
        },
    )


@configclass
class DepthCommandsCfg(CommandsCfg):
    """Depth task command configuration with odometry-bias state."""

    motion: DepthMotionCommandCfg = DepthMotionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        rsi=True,
    )


@configclass
class G1HoiDepthEnvCfg(G1HoiLearningEnvCfg):
    scene: DepthSceneCfg = DepthSceneCfg(num_envs=4096, env_spacing=4.0)
    commands: DepthCommandsCfg = DepthCommandsCfg()
    observations: DepthObservationsCfg = DepthObservationsCfg()
    events: DepthEventsCfg = DepthEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
