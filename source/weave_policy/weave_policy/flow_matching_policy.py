"""Reference normalization -> GR00T flow objective -> DiT -> velocity decoder."""

import json
import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

from .config import ReferencePolicyConfig
from .data.normalization import ReferenceNormalizer
from .data.reference_codec import sanitize_reference
from .models.dit import ReferenceDiT
from .models.reference_decoder import ReferenceDecoder
from .models.reference_encoder import ReferenceEncoder


class FlowMatchingPolicy(nn.Module):
    def __init__(self, config, statistics):
        super().__init__()
        self.config = config
        self.statistics = statistics
        self.normalizer = ReferenceNormalizer(statistics, config.normalization, config.clip_outliers)
        self.image_norm = nn.LayerNorm(config.image_dim)
        self.reference_encoder = ReferenceEncoder(config.reference_dim, config.width, config.num_embodiments)
        self.position_embedding = nn.Embedding(config.horizon, config.width)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.dit = ReferenceDiT(config)
        self.reference_decoder = ReferenceDecoder(config.width, config.reference_dim, config.num_embodiments)
        self.beta = Beta(torch.tensor(config.noise_beta_alpha), torch.tensor(config.noise_beta_beta))

    @property
    def device(self):
        return self.position_embedding.weight.device

    def _images(self, tokens):
        if tokens.shape[1:] != (self.config.image_tokens, self.config.image_dim):
            raise ValueError("Expected image tokens [B,257,768]")
        if not tokens.is_floating_point() or not torch.isfinite(tokens).all():
            raise ValueError("Expected finite floating point DINOv2 tokens")
        return self.image_norm(tokens.to(self.device, dtype=self.image_norm.weight.dtype))

    def _categories(self, batch, category_ids):
        if category_ids is None:
            return torch.zeros(batch, dtype=torch.long, device=self.device)
        category_ids = category_ids.to(self.device)
        if category_ids.shape != (batch,) or category_ids.dtype != torch.long:
            raise ValueError("Expected embodiment IDs int64[B]")
        if (category_ids < 0).any() or (category_ids >= self.config.num_embodiments).any():
            raise ValueError("Unknown embodiment ID")
        return category_ids

    def predict_velocity(self, noisy_reference, timestep, conditioned_images, category_ids):
        dtype = self.position_embedding.weight.dtype
        tokens = self.reference_encoder(noisy_reference.to(dtype), timestep, category_ids)
        position_ids = torch.arange(self.config.horizon, device=self.device)
        tokens = tokens + self.position_embedding(position_ids)[None]
        output = self.dit(tokens, conditioned_images, timestep)
        return self.reference_decoder(output, category_ids).float()

    def forward(self, image_tokens, reference, reference_mask, category_ids=None, *, noise=None, time=None):
        images = self._images(image_tokens)
        shape = (len(images), self.config.horizon, self.config.reference_dim)
        if reference.shape != shape or reference_mask.shape != shape:
            raise ValueError("Expected reference and mask [B,10,125]")
        reference = reference.to(self.device, dtype=torch.float32)
        mask = reference_mask.to(self.device, dtype=torch.float32)
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Expected a binary reference mask")
        clean = self.normalizer.normalize(reference)
        noise = torch.randn_like(clean) if noise is None else noise.to(self.device, dtype=torch.float32)
        if noise.shape != shape or not torch.isfinite(noise).all():
            raise ValueError("Invalid flow noise")
        if time is None:
            time = (1 - self.beta.sample((len(images),))).to(self.device) * self.config.noise_s
        else:
            time = time.to(self.device, dtype=torch.float32)
        if time.shape != (len(images),) or not torch.isfinite(time).all() or (time < 0).any() or (time >= 1).any():
            raise ValueError("Expected finite flow time [B] in [0,1)")
        tau = time[:, None, None]
        noisy = (1 - tau) * noise + tau * clean
        target = clean - noise
        timestep = (time * self.config.num_timestep_buckets).long()
        prediction = self.predict_velocity(noisy, timestep, images, self._categories(len(images), category_ids))
        # Expose the differentiable numerator and integer denominator separately.
        # The trainer normalizes over all ranks and accumulation microbatches.
        loss_sum = (F.mse_loss(prediction.float(), target.float(), reduction="none") * mask).sum()
        valid_count = reference_mask.to(device=self.device, dtype=torch.int64).sum()
        loss = loss_sum / (valid_count.to(loss_sum.dtype) + 1e-6)
        return {
            "loss": loss,
            "loss_sum": loss_sum,
            "valid_count": valid_count,
            "pred_velocity": prediction,
            "target_velocity": target,
            "time": time,
        }

    @torch.no_grad()
    def sample_normalized(self, image_tokens, num_steps=None, category_ids=None, *, noise=None, generator=None):
        if self.training:
            raise RuntimeError("Call policy.eval() before sampling so dropout is disabled")
        images = self._images(image_tokens)
        steps = self.config.num_inference_steps if num_steps is None else num_steps
        if type(steps) is not int or steps <= 0:
            raise ValueError("Inference steps must be a positive integer")
        shape = (len(images), self.config.horizon, self.config.reference_dim)
        if noise is None:
            value = torch.randn(shape, device=self.device, generator=generator)
        else:
            value = noise.to(self.device).float()
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError("Invalid initial flow noise")
        ids = self._categories(len(images), category_ids)
        for step in range(steps):
            timestep = torch.full(
                (len(images),),
                int(step / steps * self.config.num_timestep_buckets),
                device=self.device,
                dtype=torch.long,
            )
            value = value + self.predict_velocity(value, timestep, images, ids) / steps
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite flow prediction")
        return value

    @torch.no_grad()
    def sample(self, image_tokens, num_steps=None, category_ids=None, **kwargs):
        normalized = self.sample_normalized(image_tokens, num_steps, category_ids, **kwargs)
        return sanitize_reference(self.normalizer.unnormalize(normalized))

    def save_checkpoint(self, path, metadata, training_state=None):
        """Atomically publish a new checkpoint directory; never overwrite one."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        try:
            (temporary / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
            (temporary / "normalization.json").write_text(json.dumps(self.statistics, indent=2))
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2))
            torch.save(self.state_dict(), temporary / "model.pt")
            if training_state is not None:
                torch.save(training_state, temporary / "training_state.pt")
            temporary.rename(path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        path = Path(path)
        config = ReferencePolicyConfig.from_dict(json.loads((path / "config.json").read_text()))
        statistics = json.loads((path / "normalization.json").read_text())
        policy = cls(config, statistics)
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        policy.load_state_dict(state, strict=True)
        return policy.to(device).eval(), json.loads((path / "metadata.json").read_text())
