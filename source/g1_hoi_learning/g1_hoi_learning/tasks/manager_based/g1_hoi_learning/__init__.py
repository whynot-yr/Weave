# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv

from g1_hoi_learning.objects.object_cfg import OBJECT_CFG_BY_NAME

from . import agents


def _make_env(cfg, **kwargs):
    """Resolve scene.object from motion_file's object_name, then construct the env.
    """
    motion_file = cfg.commands.motion.motion_file
    if not motion_file:
        raise ValueError(
            "commands.motion.motion_file must be set. Pass it via Hydra CLI override:\n"
            "  env.commands.motion.motion_file=./data/example_data/smallbox.npz"
        )
    object_name = str(np.load(motion_file, allow_pickle=True)["object_name"])
    cfg.scene.object = OBJECT_CFG_BY_NAME[object_name].replace(prim_path="{ENV_REGEX_NS}/Object")
    print(f"[INFO]: Resolved scene.object = {object_name} (from {motion_file})")
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
