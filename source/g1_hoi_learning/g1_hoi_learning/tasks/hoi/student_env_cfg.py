"""DAgger-only environment extension. Original teacher task defaults remain unchanged."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from .env_cfg import G1HoiLearningEnvCfg, ObservationsCfg
from .mdp.depth import head_depth


@configclass
class HeadDepthCfg(ObservationGroupCfg):
    image = ObservationTermCfg(func=head_depth)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = 0  # one NCHW term; do not flatten image dimensions


@configclass
class StudentObservationsCfg(ObservationsCfg):
    head_depth: HeadDepthCfg = HeadDepthCfg()


@configclass
class G1StudentWeaveEnvCfg(G1HoiLearningEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.head_camera.width = 128
        self.scene.head_camera.height = 128
        self.scene.head_camera.data_types = ["depth"]
        self.scene.head_camera.update_period = 0.0
        self.num_rerenders_on_reset = 1
