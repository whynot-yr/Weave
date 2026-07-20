# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Depth-perception distillation."""

import gymnasium as gym

from g1_hoi_learning.tasks.hoi import _make_env

from . import agents

gym.register(
    id="G1-Inspire-HOI-Depth-v0",
    entry_point=_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1HoiDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_cfg:DepthRunnerCfg",
    },
)
