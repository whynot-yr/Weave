"""Depth-perception env."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass

from g1_hoi_learning.tasks.hoi.env_cfg import EventCfg, G1HoiLearningEnvCfg, G1HoiLearningSceneCfg, ObservationsCfg

from . import mdp
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
            width=192,
            height=108,
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
    class DepthCfg(ObsGroup):
        depth = ObsTerm(
            func=mdp.object_depth_b,
            params={
                "sensor_name": "depth_cam",
                "max_dist": 3.0,
                "min_z": 0.25,
                "noise_k_range": (0.005, 0.015),
                "dropout_prob": 0.05,
                "defm_size": 512,
                "debug_vis": False,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()


@configclass
class DepthEventsCfg(EventCfg):
    """hoi events + per-reset domain randomization of the depth camera's extrinsics / intrinsics."""

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
class G1HoiDepthEnvCfg(G1HoiLearningEnvCfg):
    scene: DepthSceneCfg = DepthSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: DepthObservationsCfg = DepthObservationsCfg()
    events: DepthEventsCfg = DepthEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
