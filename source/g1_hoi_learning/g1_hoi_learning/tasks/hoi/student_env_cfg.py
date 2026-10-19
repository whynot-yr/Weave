"""DAgger-only environment extension. Original teacher task defaults remain unchanged."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from .env_cfg import G1HoiLearningEnvCfg, ObservationsCfg
from .mdp.depth import head_depth
from .mdp.observations import goal_object_pose, goal_robot_pose


@configclass
class GoalRobotPoseCfg(ObservationGroupCfg):
    pose = ObservationTermCfg(func=goal_robot_pose, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GoalObjectPoseCfg(ObservationGroupCfg):
    pose = ObservationTermCfg(func=goal_object_pose, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


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
    goal_robot_pose: GoalRobotPoseCfg = GoalRobotPoseCfg()
    goal_object_pose: GoalObjectPoseCfg = GoalObjectPoseCfg()


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
