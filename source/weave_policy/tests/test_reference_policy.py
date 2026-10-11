import json
import types
from dataclasses import replace

import h5py
import pytest
import torch
from weave_policy.config import ReferencePolicyConfig
from weave_policy.data.normalization import ReferenceNormalizer, compute_statistics, load_statistics
from weave_policy.data.reference_codec import (
    ReferenceBuffer,
    matrix_from_quat,
    matrix_from_rot6d,
    ppo_reference_groups,
    reanchor_reference,
    sanitize_reference,
)
from weave_policy.data.reference_dataset import ReferenceWindowDataset
from weave_policy.data.split import load_split, make_split
from weave_policy.flow_matching_policy import FlowMatchingPolicy
from weave_policy.predictor import ReferencePredictor


def tiny_config():
    return replace(ReferencePolicyConfig(), width=32, num_heads=2, head_dim=16, num_layers=2, dropout=0)


def test_windows_mask_anchor_and_split(source, tmp_path):
    split = make_split(source)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    assert load_split(path, source) == split
    assert set(split["train_episodes"]) in ({0, 1}, {2, 3})
    ds = ReferenceWindowDataset(source, split["train_episodes"])
    first, last = ds.numeric_item(0), ds.numeric_item(11)
    assert first["reference_mask"][:, 0].tolist() == [True] * 3 + [False] * 2
    torch.testing.assert_close(first["reference"][:3, 53], torch.tensor([0.0, 0.5, 1.0]))
    assert last["reference_mask"].sum() == 125
    assert last["reference"][0, 53] == 0
    stats = compute_statistics(ds)
    assert stats["valid_target_rows"] == 2 * (12 + 7 + 2)
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    load_statistics(path, ds)
    with pytest.raises(ValueError, match="split"):
        load_statistics(path, ReferenceWindowDataset(source, split["val_episodes"]))
    split["train_episodes"], split["val_episodes"] = [0, 2], [1, 3]
    (tmp_path / "split.json").write_text(json.dumps(split))
    with pytest.raises(ValueError, match="leaked"):
        load_split(tmp_path / "split.json", source)


@pytest.mark.parametrize("mode", ["minmax", "minmax_extrema", "meanstd"])
def test_normalization_contacts_and_constant(statistics, mode):
    statistics["statistics"]["min"][0] = statistics["statistics"]["max"][0] = 0.3
    statistics["statistics"]["q01"][0] = statistics["statistics"]["q99"][0] = 0.3
    statistics["statistics"]["mean"][0], statistics["statistics"]["std"][0] = 0.3, 0
    norm = ReferenceNormalizer(statistics, mode)
    value = torch.full((1, 5, 125), 0.5)
    value[..., 0] = 0.3
    value[..., 71:] = torch.arange(54) % 3 - 1
    normalized = norm.normalize(value)
    assert (normalized[..., 0] == 0).all()
    torch.testing.assert_close(normalized[..., 71:], value[..., 71:])
    torch.testing.assert_close(norm.unnormalize(normalized), value)


def test_flow_objective_mask_and_euler(statistics):
    policy = FlowMatchingPolicy(tiny_config(), statistics)
    images, reference = torch.randn(2, 257, 768), torch.rand(2, 5, 125)
    noise, time = torch.randn_like(reference), torch.tensor([0.25, 0.75])
    mask = torch.zeros_like(reference)
    mask[:, :2] = 1
    captured = {}

    def constant_velocity(self, noisy, timestep, images, ids):
        captured.update(noisy=noisy, timestep=timestep)
        return torch.zeros_like(noisy)

    policy.predict_velocity = types.MethodType(constant_velocity, policy)
    result = policy(images, reference, mask, noise=noise, time=time)
    expected = policy.normalizer.normalize(reference)
    torch.testing.assert_close(captured["noisy"], (1 - time[:, None, None]) * noise + time[:, None, None] * expected)
    torch.testing.assert_close(captured["timestep"], torch.tensor([250, 750]))
    torch.testing.assert_close(result["target_velocity"], expected - noise)
    torch.testing.assert_close(result["loss"], ((expected - noise).square() * mask).sum() / (mask.sum() + 1e-6))
    policy.predict_velocity = types.MethodType(lambda self, x, *args: expected - noise, policy)
    policy.eval()
    torch.testing.assert_close(policy.sample_normalized(images, noise=noise), expected, atol=5e-7, rtol=1e-5)


def test_training_checkpoint_and_predictor(statistics, tmp_path):
    policy = FlowMatchingPolicy(tiny_config(), statistics)
    images, refs = torch.randn(1, 257, 768), torch.randn(1, 5, 125)
    result = policy(images, refs, torch.ones_like(refs))
    result["loss"].backward()
    assert policy.reference_encoder.W1.W.grad is not None
    assert policy.dit.transformer_blocks[0].attn1.to_k.weight.grad is not None
    assert policy.reference_decoder.layer2.W.grad is not None
    policy.eval()
    noise = torch.randn_like(refs)
    expected = policy.sample(images, noise=noise)
    checkpoint = tmp_path / "checkpoint"
    policy.save_checkpoint(checkpoint, {"encoder": {"encoder_id": "test"}})
    restored, _ = FlowMatchingPolicy.from_checkpoint(checkpoint)
    torch.testing.assert_close(restored.sample(images, noise=noise), expected, atol=0, rtol=0)
    predictor = ReferencePredictor(checkpoint)
    torch.testing.assert_close(
        predictor.predict_tokens(images, "test", seed=3),
        predictor.predict_tokens(images, "test", seed=3),
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError, match="encoder"):
        predictor.predict_tokens(images, "wrong")
    with pytest.raises(FileExistsError):
        policy.save_checkpoint(checkpoint, {})


def test_rotation_reanchor_and_ppo_layout(source):
    ds = ReferenceWindowDataset(source, [0])
    refs = ds.numeric_item(0)["reference"]
    pos, quat = torch.zeros(3), torch.tensor([1.0, 0, 0, 0])
    new_pos, new_quat = torch.tensor([0.4, 0.3, 0.1]), torch.tensor([0.9238795, 0, 0, 0.3826834])
    selected = refs[:5]
    actual = reanchor_reference(selected, pos, quat, new_pos, new_quat)
    inverse = matrix_from_quat(new_quat).T
    torch.testing.assert_close(actual[..., 53:56], (inverse @ (selected[..., 53:56] - new_pos).T).T)
    projected = matrix_from_rot6d(sanitize_reference(torch.zeros(5, 125))[..., 56:62])
    torch.testing.assert_close(projected.transpose(-1, -2) @ projected, torch.eye(3).expand(5, 3, 3))
    assert (torch.linalg.det(projected) > 0.999).all()
    groups = ppo_reference_groups(selected)
    assert groups["ref_motion_body"].shape == (310,)
    torch.testing.assert_close(groups["ref_motion_body"][:265], selected[:, :53].flatten())
    assert groups["ref_motion_object"].shape == (315,)
    buffer = ReferenceBuffer(range(0, 25, 5))
    with pytest.raises(ValueError, match="prediction ends at 20"):
        ReferenceBuffer(range(0, 25, 5), replan_steps=2)
    buffer.install(refs, pos, quat, 0)
    torch.testing.assert_close(buffer.groups(pos, quat, 0)["ref_motion_body"][0], groups["ref_motion_body"])
    with pytest.raises(RuntimeError, match="expired"):
        buffer.groups(pos, quat, 1)


def test_cache_provenance(source, tmp_path):
    from weave_policy.data.reference_dataset import inspect_source

    cache = tmp_path / "features.h5"
    with h5py.File(cache, "w") as file:
        file.attrs.update(
            complete=True,
            source_id=inspect_source(source)["source_id"],
            encoder_json=json.dumps({"encoder_id": "test"}),
        )
        file.create_dataset("tokens", shape=(12, 257, 768), dtype="f2", compression="lzf")
        file["episode_index"] = [0] * 12
        file["frame_index"] = list(range(12))
    ds = ReferenceWindowDataset(source, [0], features=cache)
    assert ds[0]["image_tokens"].shape == (257, 768)
    ds.close()
    with pytest.raises(ValueError, match="missing"):
        ReferenceWindowDataset(source, [1], features=cache)
    with h5py.File(cache, "a") as file:
        file.attrs["source_id"] = "wrong"
    with pytest.raises(ValueError, match="mismatched"):
        ReferenceWindowDataset(source, [0], features=cache)
