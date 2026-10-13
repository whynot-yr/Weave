"""Strict teacher loading, actor transfer, and a depth-only replacement for object_state."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from rsl_rl.networks import EmpiricalNormalization
from torch import nn

from g1_hoi_learning.algorithms.networks import SimBa, build_group_backbone
from g1_hoi_learning.algorithms.vision import DEPTH_SIZE, DepthCNNEncoder, preprocess_depth

TEACHER_GROUPS = ("ref_motion_body", "ref_motion_object", "object_state", "robot_proprio")
VECTOR_GROUPS = ("ref_motion_body", "ref_motion_object", "robot_proprio")
STUDENT_GROUPS = ("ref_motion_body", "ref_motion_object", "head_depth", "robot_proprio")


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
        source = make_actor(spec)
        self.vector_encoders = nn.ModuleDict({g: source[0].encoders[TEACHER_GROUPS.index(g)] for g in VECTOR_GROUPS})
        self.vector_normalizers = nn.ModuleDict(
            {g: normalizer(spec["group_dims"][g], spec["normalize"]) for g in VECTOR_GROUPS}
        )
        self.depth_encoder = DepthCNNEncoder(spec["latent_dim"])
        self.backbone: SimBa = source[1]

    @torch.no_grad()
    def initialize_from_teacher(self, teacher: TeacherWeaveActor):
        if teacher.spec != self.spec:
            raise ValueError("Teacher and student actor construction contracts differ")
        offset = 0
        for name in TEACHER_GROUPS:
            width = self.spec["group_dims"][name]
            if name in VECTOR_GROUPS:
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
        return self

    def set_visual_only(self, visual_only: bool):
        self.depth_encoder.requires_grad_(True)
        self.vector_encoders.requires_grad_(not visual_only)
        self.backbone.requires_grad_(not visual_only)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        latents = {g: self.vector_encoders[g](self.vector_normalizers[g](obs[g])) for g in VECTOR_GROUPS}
        latents["head_depth"] = self.depth_encoder(obs["head_depth"])
        return self.backbone(torch.cat([latents[g] for g in STUDENT_GROUPS], dim=-1))


class StudentInference(nn.Module):
    """Export contract: three vector groups and NCHW image-plane depth in meters."""

    def __init__(self, student: StudentWeaveActor):
        super().__init__()
        self.student = copy.deepcopy(student).cpu().eval()

    def forward(self, ref_motion_body, ref_motion_object, robot_proprio, depth_m):
        return self.student(
            {
                "ref_motion_body": ref_motion_body,
                "ref_motion_object": ref_motion_object,
                "robot_proprio": robot_proprio,
                "head_depth": preprocess_depth(depth_m),
            }
        )


def export_student(student: StudentWeaveActor, path: str):
    module = StudentInference(student)
    examples = tuple(torch.zeros(1, student.spec["group_dims"][g]) for g in VECTOR_GROUPS)
    examples += (torch.ones(1, 1, DEPTH_SIZE, DEPTH_SIZE),)
    torch.jit.trace(module, examples).save(path)
