# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv

from g1_hoi_learning.objects.object_cfg import OBJECT_CFG_BY_NAME

from . import agents


def _make_env(cfg, **kwargs):
    """Resolve scene.object from motion_files' object_names.

    Builds a MultiAssetSpawnerCfg(random_choice=False) so envs are assigned objects in round-robin
    order: env i -> object (i % O), matching MotionCommand.env_object. Forces replicate_physics=False
    (required by IsaacLab when spawning heterogeneous assets across envs).
    """
    from .mdp.commands import MotionLoader

    motion_files = cfg.commands.motion.motion_files

    # Unique objects across all clips, in the SAME order MotionLoader uses, so the object spawned for
    # env i (assets_cfg[i % O]) matches that env's fixed object (env_object = arange % O).
    object_names = MotionLoader.object_names_of(motion_files)
    assets_spawn_cfg: list[sim_utils.SpawnerCfg] = []
    for name in object_names:
        if name not in OBJECT_CFG_BY_NAME:
            raise KeyError(f"Unknown object_name '{name}'; not in OBJECT_CFG_BY_NAME")
        assets_spawn_cfg.append(OBJECT_CFG_BY_NAME[name].spawn)

    cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=assets_spawn_cfg,
            random_choice=False,
            activate_contact_sensors=True,
        ),
    )
    cfg.scene.replicate_physics = False

    print(f"[INFO]: Resolved scene.object across {len(object_names)} object(s): {object_names}")
    return ManagerBasedRLEnv(cfg=cfg, **kwargs)


gym.register(
    id="G1-Inspire-HOI-v0",
    entry_point=_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_hoi_learning_env_cfg:G1HoiLearningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
