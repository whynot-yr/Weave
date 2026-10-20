"""Frozen trajectory teacher and terminal-goal student with shared live object observations."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from rsl_rl.networks import EmpiricalNormalization
from torch import nn

from g1_hoi_learning.algorithms.networks import MLPEncoder, SimBa, build_group_backbone

TEACHER_GROUPS = ("ref_motion_body", "ref_motion_object", "object_state", "robot_proprio")
STUDENT_GROUPS = ("goal_robot_pose", "goal_object_pose", "object_state", "robot_proprio")
GOAL_VECTOR_GROUPS = STUDENT_GROUPS
TRANSFER_GROUPS = ("object_state", "robot_proprio")


def student_groups(spec: dict) -> tuple[str, ...]:
    """The student uses terminal goals and the teacher's live object/proprio groups."""
    if spec.get("observation_mode") != "terminal_goal" or spec.get("perception_mode") != "object_state":
        raise ValueError("Student requires terminal_goal observations and object_state perception")
    return STUDENT_GROUPS


def student_actor_spec(teacher_spec: dict, group_dims: dict[str, int]) -> dict:
    """Independent student contract; object_state retains the teacher's full live observation."""
    if group_dims["goal_robot_pose"] <= 9 or group_dims["goal_object_pose"] != 9:
        raise ValueError("Goal inputs must contain joint angles + anchor pose, and a 9D object pose")
    for group in TRANSFER_GROUPS:
        if group_dims[group] != teacher_spec["group_dims"][group]:
            raise ValueError(f"Student and teacher {group} dimensions differ")
    spec = copy.deepcopy(teacher_spec)
    spec["observation_mode"] = "terminal_goal"
    spec["perception_mode"] = "object_state"
    spec["group_dims"] = {g: group_dims[g] for g in GOAL_VECTOR_GROUPS}
    spec["encoder_specs"] = {
        "goal_robot_pose": copy.deepcopy(teacher_spec["encoder_specs"]["ref_motion_body"]),
        "goal_object_pose": copy.deepcopy(teacher_spec["encoder_specs"]["ref_motion_object"]),
        "object_state": copy.deepcopy(teacher_spec["encoder_specs"]["object_state"]),
        "robot_proprio": copy.deepcopy(teacher_spec["encoder_specs"]["robot_proprio"]),
    }
    spec["goal_contract"] = {
        "version": 1,
        "source": "selected_clip_last_frame",
        "frame": "current_actual_robot_anchor",
        "rotation": "matrix_first_two_columns_row_major",
        "robot_fields": ["joint_pos", "anchor_pos", "anchor_rot6d"],
        "object_fields": ["object_pos", "object_rot6d"],
        "normalization": "identity",
    }
    return spec


def actor_spec(agent_cfg: dict, group_dims: dict[str, int], num_actions: int) -> dict:
    """Record an explicit, checkpoint-portable actor construction contract."""
    policy = agent_cfg["policy"]
    if policy["class_name"] != "SimBaActorCritic" or policy.get("state_dependent_std", False):
        raise ValueError("DAgger requires a state-independent SimBaActorCritic teacher")
    if tuple(agent_cfg["obs_groups"]["policy"]) != TEACHER_GROUPS:
        raise ValueError(f"Teacher policy groups must have this exact order: {TEACHER_GROUPS}")
    for name in TEACHER_GROUPS:
        if policy["encoder_hidden_dims"][name]["type"] != "mlp":
            raise ValueError(f"Expected an MLP teacher encoder for {name}")
    return {
        "group_dims": {g: group_dims[g] for g in TEACHER_GROUPS},
        "num_actions": num_actions,
        "latent_dim": policy["latent_dim"],
        "hidden_dim": policy["actor_hidden_dim"],
        "num_blocks": policy["actor_num_blocks"],
        "expansion": policy["expansion"],
        "encoder_specs": copy.deepcopy(policy["encoder_hidden_dims"]),
        "normalize": policy.get("actor_obs_normalization", False),
    }


def make_actor(spec: dict) -> nn.Sequential:
    # The original builder uses only tensor shapes; no simulator or critic is needed.
    obs = {g: torch.empty(1, spec["group_dims"][g]) for g in TEACHER_GROUPS}
    return build_group_backbone(
        obs,
        list(TEACHER_GROUPS),
        spec["encoder_specs"],
        spec["latent_dim"],
        spec["num_actions"],
        spec["hidden_dim"],
        spec["num_blocks"],
        spec["expansion"],
    )


def normalizer(dim: int, enabled: bool) -> nn.Module:
    return EmpiricalNormalization(dim).eval() if enabled else nn.Identity()


class TeacherWeaveActor(nn.Module):
    def __init__(self, spec: dict, state_dict: Mapping[str, torch.Tensor]):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.actor = make_actor(spec)
        self.normalizer = normalizer(sum(spec["group_dims"].values()), spec["normalize"])
        actor_state = {k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")}
        norm_state = {
            k.removeprefix("actor_obs_normalizer."): v
            for k, v in state_dict.items()
            if k.startswith("actor_obs_normalizer.")
        }
        self.actor.load_state_dict(actor_state, strict=True)
        self.normalizer.load_state_dict(norm_state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Teacher must remain frozen even if its caller switches an enclosing module to train mode.
        return super().train(False)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([obs[g] for g in TEACHER_GROUPS], dim=-1)
        return self.actor(self.normalizer(x))


class StudentWeaveActor(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.observation_groups = student_groups(spec)
        self.vector_groups = self.observation_groups
        encoders = {}
        for group in self.vector_groups:
            options = dict(spec["encoder_specs"][group])
            if options.pop("type") != "mlp":
                raise ValueError(f"Expected MLP student encoder for {group}")
            encoders[group] = MLPEncoder(spec["group_dims"][group], spec["latent_dim"], **options)
        self.vector_encoders = nn.ModuleDict(encoders)
        self.vector_normalizers = nn.ModuleDict(
            {
                g: normalizer(spec["group_dims"][g], spec["normalize"] and not g.startswith("goal_"))
                for g in self.vector_groups
            }
        )
        self.backbone = SimBa(
            len(self.observation_groups) * spec["latent_dim"],
            spec["num_actions"],
            spec["hidden_dim"],
            spec["num_blocks"],
            spec["expansion"],
        )

    @torch.no_grad()
    def initialize_from_teacher(self, teacher: TeacherWeaveActor):
        shared = ("num_actions", "latent_dim", "hidden_dim", "num_blocks", "expansion", "normalize")
        if any(self.spec[key] != teacher.spec[key] for key in shared) or any(
            self.spec["group_dims"][group] != teacher.spec["group_dims"][group]
            or self.spec["encoder_specs"][group] != teacher.spec["encoder_specs"][group]
            for group in TRANSFER_GROUPS
        ):
            raise ValueError("Teacher and goal student transfer contracts differ")
        offset = 0
        for name in TEACHER_GROUPS:
            width = teacher.spec["group_dims"][name]
            if name in TRANSFER_GROUPS:
                self.vector_encoders[name].load_state_dict(
                    teacher.actor[0].encoders[TEACHER_GROUPS.index(name)].state_dict(), strict=True
                )
                if self.spec["normalize"]:
                    state = teacher.normalizer.state_dict()
                    subset = {
                        key: value[:, offset : offset + width].clone() if key != "count" else value.clone()
                        for key, value in state.items()
                    }
                    self.vector_normalizers[name].load_state_dict(subset, strict=True)
            offset += width
        self.backbone.load_state_dict(teacher.actor[1].state_dict(), strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vector_normalizers.eval()
        self.vector_encoders.eval()
        return self

    def set_dagger_trainable(self):
        """Train the complete student actor while retaining fixed normalization."""
        self.vector_encoders.requires_grad_(True)
        self.vector_normalizers.requires_grad_(False)
        self.backbone.requires_grad_(True)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        latents = {g: self.vector_encoders[g](self.vector_normalizers[g](obs[g])) for g in self.vector_groups}
        return self.backbone(torch.cat([latents[g] for g in self.observation_groups], dim=-1))


class StudentInference(nn.Module):
    """Export contract: terminal robot/object goals and live object/proprio vectors."""

    def __init__(self, student: StudentWeaveActor):
        super().__init__()
        self.student = copy.deepcopy(student).cpu().eval()

    def forward(self, goal_robot_pose, goal_object_pose, object_state, robot_proprio):
        return self.student(
            {
                "goal_robot_pose": goal_robot_pose,
                "goal_object_pose": goal_object_pose,
                "object_state": object_state,
                "robot_proprio": robot_proprio,
            }
        )


def export_student(student: StudentWeaveActor, path: str):
    module = StudentInference(student)
    examples = tuple(torch.zeros(1, student.spec["group_dims"][g]) for g in student.vector_groups)
    torch.jit.trace(module, examples).save(path)
