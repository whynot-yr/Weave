"""Student observations with staged random future goals on the base HOI environment."""

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
class GoalObservationsCfg(ObservationsCfg):
    goal_robot_pose: GoalRobotPoseCfg = GoalRobotPoseCfg()
    goal_object_pose: GoalObjectPoseCfg = GoalObjectPoseCfg()


@configclass
class G1HoiGoalEnvCfg(G1HoiLearningEnvCfg):
    observations: GoalObservationsCfg = GoalObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.motion.goal_sampling_mode = "random_future"
