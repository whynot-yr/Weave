# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv

from g1_hoi_learning.objects.object_cfg import OBJECT_CFG_BY_NAME

from . import agents


def _make_env(cfg, **kwargs):
    """Resolve scene.object from motion_files' object_names.

    Builds a MultiAssetSpawnerCfg(random_choice=False) so envs are assigned
    objects in round-robin order: env i -> motion_files[i % N].
    Forces replicate_physics=False (required by IsaacLab when spawning
    heterogeneous assets across envs).
    """
    motion_files = cfg.commands.motion.motion_files

    # collect per-file object names + per-object spawn cfgs
    object_names: list[str] = []
    assets_spawn_cfg: list[sim_utils.SpawnerCfg] = []
    for f in motion_files:
        name = str(np.load(f, allow_pickle=True)["object_name"])
        if name not in OBJECT_CFG_BY_NAME:
            raise KeyError(f"Unknown object_name '{name}' from {f}; not in OBJECT_CFG_BY_NAME")
        object_names.append(name)
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
