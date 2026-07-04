# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL distillation runner cfg — SimBa student-teacher + Muon."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class MuonDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = "MuonDistillation"
    num_learning_epochs: int = 1
    learning_rate: float = 1.0e-3
    gradient_length: int = 32          # env steps accumulated per optimizer.step()
    max_grad_norm: float = 1.0
    loss_type: str = "mse"             # "mse" to start; "huber" if OOD states spike the gradient
    weight_decay: float = 0.01

@configclass
class SimBaStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    class_name: str = "SimBaStudentTeacher"

    # base fields unused by SimBa (present to satisfy MISSING in the parent cfg)
    student_hidden_dims: list[int] = []
    teacher_hidden_dims: list[int] = []
    activation: str = "relu"

    # SimBa knobs — teacher_* MUST match the Stage-1 tracker actor (2048 / 2 / 1) so its ckpt loads
    student_hidden_dim: int = 2048
    teacher_hidden_dim: int = 2048
    student_num_blocks: int = 2
    teacher_num_blocks: int = 2
    expansion: int = 1

    student_obs_normalization: bool = True
    teacher_obs_normalization: bool = True
    init_noise_std: float = 0.5
    noise_std_type: str = "scalar"

@configclass
class DistillRunnerCfg(RslRlDistillationRunnerCfg):
    class_name: str = "DistillationRunner"
    experiment_name: str = "g1_inspire_hoi_distill"
    num_steps_per_env: int = 32
    max_iterations: int = 20000
    save_interval: int = 200
    # map runner roles -> env observation groups
    obs_groups: dict = {"policy": ["policy"], "teacher": ["teacher"]}
    policy: SimBaStudentTeacherCfg = SimBaStudentTeacherCfg()
    algorithm: MuonDistillationAlgorithmCfg = MuonDistillationAlgorithmCfg()
