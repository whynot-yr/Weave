from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import EmpiricalNormalization

from g1_hoi_learning.algorithms.networks import SimBa


class SimBaStudentTeacher(StudentTeacher):
    """``StudentTeacher`` whose student and teacher are both SimBa backbones.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = True,
        teacher_obs_normalization: bool = True,
        student_hidden_dim: int = 512,
        teacher_hidden_dim: int = 512,
        student_num_blocks: int = 2,
        teacher_num_blocks: int = 2,
        expansion: int = 4,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "SimBaStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        
        nn.Module.__init__(self)

        self.loaded_teacher = False
        self.obs_groups = obs_groups

        num_student_obs = 0
        for g in obs_groups["policy"]:
            assert len(obs[g].shape) == 2, "SimBaStudentTeacher only supports 1D observations."
            num_student_obs += obs[g].shape[-1]
        num_teacher_obs = 0
        for g in obs_groups["teacher"]:
            assert len(obs[g].shape) == 2, "SimBaStudentTeacher only supports 1D observations."
            num_teacher_obs += obs[g].shape[-1]

        # ----- student -----
        self.student = SimBa(
            input_dim=num_student_obs,
            output_dim=num_actions,
            hidden_dim=student_hidden_dim,
            num_blocks=student_num_blocks,
            expansion=expansion,
        )
        print(f"Student SimBa: {self.student}")
        self.student_obs_normalization = student_obs_normalization
        self.student_obs_normalizer = (
            EmpiricalNormalization(num_student_obs) if student_obs_normalization else nn.Identity()
        )

        # ----- teacher (frozen) -----
        self.teacher = SimBa(
            input_dim=num_teacher_obs,
            output_dim=num_actions,
            hidden_dim=teacher_hidden_dim,
            num_blocks=teacher_num_blocks,
            expansion=expansion,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        print(f"Teacher SimBa: {self.teacher}")
        self.teacher_obs_normalization = teacher_obs_normalization
        self.teacher_obs_normalizer = (
            EmpiricalNormalization(num_teacher_obs) if teacher_obs_normalization else nn.Identity()
        )

        # Action noise (state-independent)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)
