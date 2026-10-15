"""3904770 actor/critic/teacher contract adapted to the existing depth student.

The student's inputs, CNN, normalization and tracker initialization stay in
dagger.networks. RSL-RL supplies the Gaussian distribution and critic interfaces.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.modules import ActorCritic
from torch import nn

from g1_hoi_learning.algorithms.dagger.networks import StudentWeaveActor, TeacherWeaveActor, normalizer
from g1_hoi_learning.algorithms.networks import build_group_backbone


def critic_spec(agent_cfg: dict, obs) -> dict:
    policy = agent_cfg["policy"]
    groups = list(agent_cfg["obs_groups"]["critic"])
    if not groups or any(obs[g].ndim != 2 for g in groups):
        raise ValueError("Critic requires flat observation groups from the teacher configuration")
    return {
        "groups": groups,
        "group_dims": {g: obs[g].shape[-1] for g in groups},
        "encoder_specs": copy.deepcopy(policy["encoder_hidden_dims"]),
        "latent_dim": policy["latent_dim"],
        "hidden_dim": policy["critic_hidden_dim"],
        "num_blocks": policy["critic_num_blocks"],
        "expansion": policy["expansion"],
        "normalize": policy.get("critic_obs_normalization", False),
    }


class SimBaActorCriticTeacher(ActorCritic):
    """Historical PPO+BC policy with the current student as its mean actor."""

    def __init__(
        self,
        student: StudentWeaveActor,
        teacher: TeacherWeaveActor,
        spec: dict,
        teacher_state: dict,
        init_noise_std: float = 0.2,
    ):
        nn.Module.__init__(self)
        self.actor = student
        self.actor.set_dagger_trainable()
        self.teacher = teacher
        self.critic_spec = copy.deepcopy(spec)
        self.obs_groups = {"critic": spec["groups"]}
        self.actor_obs_normalization = False  # Already performed inside StudentWeaveActor.
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalization = spec["normalize"]
        obs = {g: torch.empty(1, spec["group_dims"][g]) for g in spec["groups"]}
        self.critic = build_group_backbone(
            obs,
            spec["groups"],
            spec["encoder_specs"],
            spec["latent_dim"],
            1,
            spec["hidden_dim"],
            spec["num_blocks"],
            spec["expansion"],
        )
        self.critic_obs_normalizer = normalizer(sum(spec["group_dims"].values()), spec["normalize"])
        self.critic.load_state_dict(
            {k.removeprefix("critic."): v for k, v in teacher_state.items() if k.startswith("critic.")}, strict=True
        )
        self.critic_obs_normalizer.load_state_dict(
            {
                k.removeprefix("critic_obs_normalizer."): v
                for k, v in teacher_state.items()
                if k.startswith("critic_obs_normalizer.")
            },
            strict=True,
        )
        self.state_dependent_std = False
        self.noise_std_type = "scalar"
        self.std = nn.Parameter(torch.full((student.spec["num_actions"],), init_noise_std))
        torch.distributions.Normal.set_default_validate_args(False)
        self.distribution = None
        self.teacher.requires_grad_(False)
        self.critic_obs_normalizer.requires_grad_(False)
        self.train()

    def get_actor_obs(self, obs):
        # Keep NCHW depth separate; never concatenate or normalize it a second time.
        return obs

    def update_normalization(self, obs):
        """All normalizers are teacher-initialized and fixed by design."""

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.critic_obs_normalizer.eval()
        return self

    @torch.no_grad()
    def teacher_act(self, obs):
        return self.teacher(obs)

    def forward(self, obs, actions):
        """One DDP forward contains every trainable branch used by PPO+BC."""
        self._update_distribution(obs)
        return (
            self.action_mean,
            self.action_std,
            self.evaluate(obs),
            self.get_actions_log_prob(actions),
            self.entropy,
        )

    @torch.no_grad()
    def protect_std(self):
        self.std.clamp_(min=1e-4)
