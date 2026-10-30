import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, EventTermCfg, TerminationTermCfg, RewardTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from g1_hoi_learning.assets import GROUND_PLANE_USD_PATH
from g1_hoi_learning.robots.g1_inspire import G1_INSPIRE_CFG
from g1_hoi_learning.tasks.hoi.env_cfg import ActionsCfg
from .pose_cfg import GetupPoseCfg
from . import mdp
from .mdp.rewards import WEIGHTS


@configclass
class GetupSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim_utils.GroundPlaneCfg(
        usd_path=GROUND_PLANE_USD_PATH, size=(100., 100.),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)))
    robot = G1_INSPIRE_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=800.))


@configclass
class ObservationsCfg:
    @configclass
    class ProprioCfg(ObservationGroupCfg):
        history = ObservationTermCfg(func=mdp.proprio_history)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False
    robot_proprio: ProprioCfg = ProprioCfg()


@configclass
class EventsCfg:
    reset = EventTermCfg(func=mdp.reset_supine, mode='reset')


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    joint_velocity = TerminationTermCfg(func=mdp.joint_velocity_out)
    base_velocity = TerminationTermCfg(func=mdp.base_velocity_out)


@configclass
class GetupEnvCfg(ManagerBasedRLEnvCfg):
    scene: GetupSceneCfg = GetupSceneCfg(num_envs=4096, env_spacing=3.0)
    pose: GetupPoseCfg = GetupPoseCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: dict = {name: RewardTermCfg(func=mdp.reward_term, weight=weight, params={'name': name})
                     for name, weight in WEIGHTS.items()}
    advantage_weights: tuple = (2.5, 1., .1, 1.)

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.
        self.sim.dt = .005
        self.sim.render_interval = 4
        self.sim.physx.gpu_max_rigid_patch_count = 16*2**16
        self.viewer.eye = (2., 2., 1.5)
        self.viewer.lookat = (0., 0., .4)
