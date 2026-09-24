import json

import h5py
import numpy as np
import pytest
import torch
from weave_data.dp_codec import IMAGE_KEY
from weave_data.dp_dataset import DPWindowDataset
from weave_data.dp_stats import compute_stats, load_stats
from weave_data.hdf5_converter import convert_hdf5


@pytest.fixture(params=[False, True], ids=["images", "video"])
def converted(tmp_path, request):
    pytest.importorskip("lerobot")
    source = tmp_path / "rollout.h5"
    E, T = 3, 4
    with h5py.File(source, "w") as f:
        f.attrs.update(
            schema_version="weave-rollout-v1", quaternion_order="wxyz", coordinate_frame="env_local_world_axes"
        )
        f.attrs["metadata_json"] = json.dumps(
            {
                "joint_names": [f"j{i}" for i in range(53)],
                "action_names": [f"j{i}" for i in range(39)] + ["j43", "j48"],
                "body_names": ["pelvis"] + [f"b{i}" for i in range(53)],
                "fps": 50,
            }
        )
        f["episodes/length"] = [T, T, T]
        f["episodes/success"] = [True, False, True]
        f["episodes/object_name"] = np.array(["lamp"] * E, dtype=h5py.string_dtype())
        f["episodes/clip_name"] = np.array(["motion_a_v00", "motion_b_v00", "motion_c_v00"], dtype=h5py.string_dtype())
        f["frame/valid"] = np.ones((E, T), dtype=bool)
        f["frame/sim_time"] = np.tile(np.arange(T) / 50, (E, 1))
        f["camera/head/rgb"] = np.full((E, T, 224, 224, 3), 127, dtype=np.uint8)
        for prefix in ("robot", "reference"):
            f[f"{prefix}/joint_pos"] = np.tile(np.arange(53, dtype=np.float32), (E, T, 1))
            f[f"{prefix}/joint_vel"] = np.ones((E, T, 53), dtype=np.float32)
            pos = np.zeros((E, T, 54, 3), dtype=np.float32)
            pos[..., 0] = np.arange(T)[None, :, None]
            f[f"{prefix}/body_pos"] = pos
            q = np.zeros((E, T, 54, 4), dtype=np.float32)
            q[..., 0] = 1
            f[f"{prefix}/body_quat"] = q
        f["reference/object_pos"] = np.zeros((E, T, 3), dtype=np.float32)
        f["reference/object_quat"] = np.tile([1.0, 0.0, 0.0, 0.0], (E, T, 1)).astype(np.float32)
        f["reference/contact_label"] = np.tile(np.arange(54) % 3 - 1, (E, T, 1)).astype(np.int8)
    output = tmp_path / "converted"
    convert_hdf5(source, output, "weave/test", use_videos=request.param)
    return output


def test_window_dataset_and_stats(converted, tmp_path):
    dataset = DPWindowDataset(converted, [0, 2], horizon=40)
    assert len(dataset) == 8
    first = dataset[0]
    assert first[IMAGE_KEY].shape == (1, 3, 224, 224)
    assert first["observation.state"].shape == (1, 88)
    assert first["action"].shape == (40, 125)
    assert first["action_is_pad"].tolist() == [False] * 4 + [True] * 36
    torch.testing.assert_close(first["action"][:4, 53], torch.arange(4).float())
    end = dataset.numeric_item(3)
    assert end["action_is_pad"].sum() == 39
    assert end["action"][:, 53].eq(0).all()
    assert dataset.numeric_item(4)["action_is_pad"].sum() == 36
    stats = compute_stats(dataset)
    # Per episode: valid counts 4+3+2+1. Padding must not enter statistics.
    assert stats["stats"]["action"]["count"] == [20]
    assert stats["stats"]["action"]["min"][71:] == [-1] * 54
    assert stats["stats"]["action"]["max"][71:] == [1] * 54
    output = tmp_path / "stats.json"
    output.write_text(json.dumps(stats))
    load_stats(output, dataset)
    with pytest.raises(ValueError, match="episodes"):
        load_stats(output, DPWindowDataset(converted, [0]))
    with pytest.raises(ValueError, match="horizon"):
        load_stats(output, DPWindowDataset(converted, [0, 2], horizon=8))


def test_lerobot_loss_checkpoint_and_predictor(converted, tmp_path):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from weave_data.dp_policy import DPReferencePredictor

    dataset = DPWindowDataset(converted, [0])
    payload = compute_stats(dataset)
    stats = {k: {name: torch.tensor(v) for name, v in entry.items()} for k, entry in payload["stats"].items()}
    config = DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (88,)),
            IMAGE_KEY: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (125,))},
        device="cpu",
        n_obs_steps=1,
        horizon=40,
        n_action_steps=40,
        down_dims=(32, 64, 128),
        pretrained_backbone_weights=None,
        num_inference_steps=2,
        do_mask_loss_for_padding=True,
    )
    policy = DiffusionPolicy(config)
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    assert set(batch) == {IMAGE_KEY, "observation.state", "action", "action_is_pad"}
    loss, _ = policy(pre(batch))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    policy.eval()
    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    (checkpoint / "weave_protocol.json").write_text(json.dumps(dataset.protocol))
    inputs = {k: v for k, v in batch.items() if k in {IMAGE_KEY, "observation.state"}}
    with torch.no_grad():
        torch.manual_seed(7)
        expected = post(policy.predict_action_chunk(pre(inputs)))
    predictor = DPReferencePredictor(checkpoint, "cpu")
    torch.manual_seed(7)
    actual = predictor.predict(
        torch.full((2, 224, 224, 3), 127, dtype=torch.uint8),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
        torch.arange(53).float().expand(2, -1),
        torch.ones(2, 53),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
    )
    assert actual.shape == (2, 40, 125)
    assert torch.isfinite(actual).all()
    # Lossy video can change pixel values: compare exact inference only for PNG.
    if dataset.raw.meta.features[IMAGE_KEY]["dtype"] == "image":
        torch.testing.assert_close(actual, expected)
