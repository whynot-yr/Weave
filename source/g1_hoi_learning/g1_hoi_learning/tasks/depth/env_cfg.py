"""Depth-perception env.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from g1_hoi_learning.tasks.hoi.env_cfg import G1HoiLearningEnvCfg, ObservationsCfg

from . import mdp


@configclass
class DepthObservationsCfg(ObservationsCfg):
    """Raycast depth observation group."""

    @configclass
    class DepthCfg(ObsGroup):
        """Head D435i raycast depth image."""

        depth = ObsTerm(
            func=mdp.object_depth_b,
            params={
                "cam_pos": (0.0576235, 0.01753, 0.42987),  # D435i on torso_link (URDF d435_joint)
                "cam_rpy": (0.0, 0.8307767239493009, 0.0),  # pitch ~47.6 deg down
                "fov_deg": (86.0, 57.0),
                "resolution": (80, 60),  # 640x480 downsampled 8x (MLP-tractable)
                "max_dist": 5.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()


@configclass
class G1HoiDepthEnvCfg(G1HoiLearningEnvCfg):
    observations: DepthObservationsCfg = DepthObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
