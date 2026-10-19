"""Frozen trajectory teacher and terminal-goal depth student, with legacy inference support."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from rsl_rl.networks import EmpiricalNormalization
from torch import nn

from g1_hoi_learning.algorithms.networks import MLPEncoder, SimBa, build_group_backbone
from g1_hoi_learning.algorithms.vision import DEPTH_SIZE, DepthCNNEncoder, preprocess_depth

TEACHER_GROUPS = ("ref_motion_body", "ref_motion_object", "object_state", "robot_proprio")
VECTOR_GROUPS = ("ref_motion_body", "ref_motion_object", "robot_proprio")
STUDENT_GROUPS = ("ref_motion_body", "ref_motion_object", "head_depth", "robot_proprio")
GOAL_VECTOR_GROUPS = ("goal_robot_pose", "goal_object_pose", "robot_proprio")
GOAL_STUDENT_GROUPS = ("goal_robot_pose", "goal_object_pose", "head_depth", "robot_proprio")


def student_groups(spec: dict) -> tuple[str, ...]:
    """Specs without a mode belong to the original reference-window student."""
    mode = spec.get("observation_mode", "reference")
    if mode == "terminal_goal":
        return GOAL_STUDENT_GROUPS
    if mode == "reference":
        return STUDENT_GROUPS
    raise ValueError(f"Unknown student observation mode: {mode}")


def student_actor_spec(teacher_spec: dict, group_dims: dict[str, int]) -> dict:
    """Independent student contract; no temporal references or contact labels enter its actor."""
    if group_dims["goal_robot_pose"] <= 9 or group_dims["goal_object_pose"] != 9:
        raise ValueError("Goal inputs must contain joint angles + anchor pose, and a 9D object pose")
    if group_dims["robot_proprio"] != teacher_spec["group_dims"]["robot_proprio"]:
        raise ValueError("Student and teacher proprioception dimensions differ")
    spec = copy.deepcopy(teacher_spec)
    spec["observation_mode"] = "terminal_goal"
    spec["group_dims"] = {g: group_dims[g] for g in GOAL_VECTOR_GROUPS}
    spec["encoder_specs"] = {
        "goal_robot_pose": copy.deepcopy(teacher_spec["encoder_specs"]["ref_motion_body"]),
        "goal_object_pose": copy.deepcopy(teacher_spec["encoder_specs"]["ref_motion_object"]),
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
        self.vector_groups = tuple(g for g in self.observation_groups if g != "head_depth")
        self.goal_conditioned = spec.get("observation_mode") == "terminal_goal"
        if self.goal_conditioned:
            encoders = {}
            for group in self.vector_groups:
                options = dict(spec["encoder_specs"][group])
                if options.pop("type") != "mlp":
                    raise ValueError(f"Expected MLP student encoder for {group}")
                encoders[group] = MLPEncoder(spec["group_dims"][group], spec["latent_dim"], **options)
            self.vector_encoders = nn.ModuleDict(encoders)
            backbone = SimBa(
                len(self.observation_groups) * spec["latent_dim"],
                spec["num_actions"],
                spec["hidden_dim"],
                spec["num_blocks"],
                spec["expansion"],
            )
        else:
            source = make_actor(spec)
            self.vector_encoders = nn.ModuleDict(
                {g: source[0].encoders[TEACHER_GROUPS.index(g)] for g in self.vector_groups}
            )
            backbone = source[1]
        self.vector_normalizers = nn.ModuleDict(
            {
                g: normalizer(spec["group_dims"][g], spec["normalize"] and not g.startswith("goal_"))
                for g in self.vector_groups
            }
        )
        self.depth_encoder = DepthCNNEncoder(spec["latent_dim"])
        self.backbone: SimBa = backbone

    @torch.no_grad()
    def initialize_from_teacher(self, teacher: TeacherWeaveActor):
        if self.goal_conditioned:
            shared = ("num_actions", "latent_dim", "hidden_dim", "num_blocks", "expansion", "normalize")
            if any(self.spec[key] != teacher.spec[key] for key in shared) or (
                self.spec["group_dims"]["robot_proprio"] != teacher.spec["group_dims"]["robot_proprio"]
                or self.spec["encoder_specs"]["robot_proprio"] != teacher.spec["encoder_specs"]["robot_proprio"]
            ):
                raise ValueError("Teacher and goal student transfer contracts differ")
            transfer_groups = ("robot_proprio",)
        elif teacher.spec == self.spec:
            transfer_groups = VECTOR_GROUPS
        else:
            raise ValueError("Teacher and student actor construction contracts differ")
        offset = 0
        for name in TEACHER_GROUPS:
            width = teacher.spec["group_dims"][name]
            if name in transfer_groups:
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
        self.depth_encoder.requires_grad_(True)
        self.vector_encoders.requires_grad_(True)
        self.vector_normalizers.requires_grad_(False)
        self.backbone.requires_grad_(True)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        latents = {g: self.vector_encoders[g](self.vector_normalizers[g](obs[g])) for g in self.vector_groups}
        latents["head_depth"] = self.depth_encoder(obs["head_depth"])
        return self.backbone(torch.cat([latents[g] for g in self.observation_groups], dim=-1))


class StudentInference(nn.Module):
    """Export contract: three vector groups and NCHW image-plane depth in meters."""

    def __init__(self, student: StudentWeaveActor):
        super().__init__()
        self.student = copy.deepcopy(student).cpu().eval()

    def forward(self, goal_robot_pose, goal_object_pose, robot_proprio, depth_m):
        return self.student(
            {
                self.student.vector_groups[0]: goal_robot_pose,
                self.student.vector_groups[1]: goal_object_pose,
                "robot_proprio": robot_proprio,
                "head_depth": preprocess_depth(depth_m),
            }
        )


def export_student(student: StudentWeaveActor, path: str):
    module = StudentInference(student)
    examples = tuple(torch.zeros(1, student.spec["group_dims"][g]) for g in student.vector_groups)
    examples += (torch.ones(1, 1, DEPTH_SIZE, DEPTH_SIZE),)
    torch.jit.trace(module, examples).save(path)
