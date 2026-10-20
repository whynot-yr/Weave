"""DAgger-only environment extension. Original teacher task defaults remain unchanged."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from .env_cfg import G1HoiLearningEnvCfg, ObservationsCfg
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
class StudentObservationsCfg(ObservationsCfg):
    goal_robot_pose: GoalRobotPoseCfg = GoalRobotPoseCfg()
    goal_object_pose: GoalObjectPoseCfg = GoalObjectPoseCfg()


@configclass
class G1StudentWeaveEnvCfg(G1HoiLearningEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.head_camera = None
        self.num_rerenders_on_reset = 0
