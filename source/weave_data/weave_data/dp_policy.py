"""Checkpoint inference helper; deliberately does not modify the PPO runtime."""

import json
from pathlib import Path

import torch

from .dp_codec import IMAGE_KEY, PROTOCOL, encode_state


class DPReferencePredictor:
    """Predict 40 local reference tokens from one RGB/state observation.

    The caller owns execution/replanning and stores the actual pelvis anchor
    used at prediction time. Use reanchor_reference before building PPO groups
    at subsequent control steps. This is not an actuator command interface.
    """

    def __init__(self, checkpoint, device="cuda"):
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        checkpoint = Path(checkpoint)
        self.protocol = json.loads((checkpoint / "weave_protocol.json").read_text())
        if self.protocol["protocol"] != PROTOCOL:
            raise ValueError("Incompatible WEAVE reference protocol")
        config = DiffusionConfig.from_pretrained(checkpoint)
        if config.n_obs_steps != 1 or config.horizon != 40 or config.n_action_steps != 40:
            raise ValueError("Expected a single-observation, full 40-token WEAVE checkpoint")
        config.device = device
        config.pretrained_backbone_weights = None
        self.policy = DiffusionPolicy.from_pretrained(checkpoint, config=config).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=str(checkpoint), preprocessor_overrides={"device_processor": {"device": device}}
        )

    @torch.inference_mode()
    def predict(self, rgb, pelvis_quat, joint_pos, joint_vel, initial_pelvis_quat):
        """Batched RGB uint8[B,224,224,3] and actual robot tensors; returns [B,40,125].

        Output is unnormalized and on CPU. Initial pelvis quaternion must be
        retained across replans and refreshed only on episode reset.
        """
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (224, 224, 3):
            raise ValueError("Expected uint8 RGB [B,224,224,3]")
        if joint_pos.shape != (len(rgb), 53) or joint_vel.shape != joint_pos.shape:
            raise ValueError("Expected actual full joint tensors [B,53] in the recorded joint-name order")
        if pelvis_quat.shape != (len(rgb), 4) or initial_pelvis_quat.shape != pelvis_quat.shape:
            raise ValueError("Expected current and initial pelvis quaternions [B,4]")
        state = encode_state(
            pelvis_quat, joint_pos, joint_vel, initial_pelvis_quat, self.protocol["active_joint_indices"]
        )
        batch = {"observation.state": state[:, None], IMAGE_KEY: rgb.permute(0, 3, 1, 2).float()[:, None] / 255}
        self.policy.reset()
        action = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        if action.shape != (len(rgb), 40, 125):
            raise RuntimeError(f"Unexpected reference shape: {action.shape}")
        return action
