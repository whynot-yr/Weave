"""Exercise split/stat/train/resume/infer entry points as separate processes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DistributedSampler
from weave_policy.data.normalization import compute_statistics
from weave_policy.data.reference_dataset import ReferenceWindowDataset, inspect_source
from weave_policy.data.split import make_split

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/reference_policy"


def run(script, *args):
    environment = os.environ | {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable",
))])
def test_training_resume_and_inference(source, tmp_path, device):
    split = make_split(source)
    split_path, stats_path, cache = tmp_path / "split.json", tmp_path / "stats.json", tmp_path / "cache.h5"
    run("split_dataset.py", source, split_path)
    run("compute_stats.py", source, stats_path, "--split", split_path)
    ds = ReferenceWindowDataset(source, split["train_episodes"])
    assert json.loads(stats_path.read_text()) == json.loads(json.dumps(compute_statistics(ds)))
    with h5py.File(cache, "w") as f:
        f.attrs.update(
            complete=True,
            source_id=inspect_source(source)["source_id"],
            encoder_json=json.dumps({"encoder_id": "test-fixture"}),
        )
        f.create_dataset("tokens", shape=(48, 257, 768), dtype="f2", compression="lzf")
        f["episode_index"] = np.repeat(np.arange(4), 12)
        f["frame_index"] = np.tile(np.arange(12), 4)
    args = [
        "--source",
        source,
        "--split",
        split_path,
        "--stats",
        stats_path,
        "--features",
        cache,
        "--device",
        device,
        "--workers",
        "0",
        "--steps",
        "2",
        "--batch-size",
        "2",
        "--save-every",
        "1",
        "--eval-every",
        "1",
        "--smoke",
    ]
    if device.startswith("cuda"):
        args += ["--amp", "--workers", "1"]
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    run("train.py", *args, "--output", continuous)
    run("train.py", *args, "--output", resumed, "--resume", continuous / "step_00000001")
    expected = torch.load(continuous / "step_00000002/model.pt", weights_only=True)
    actual = torch.load(resumed / "step_00000002/model.pt", weights_only=True)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    result = tmp_path / "prediction.npz"
    run("infer.py", "--checkpoint", resumed / "step_00000002", "--feature-cache", cache, "--output", result)
    with np.load(result) as f:
        assert f["reference"].shape == (1, 5, 125)
        assert f["ref_robot"].shape == (1, 5, 62)
        assert f["ref_object"].shape == (1, 5, 63)
        assert np.isfinite(f["reference"]).all()


def test_train_only_uses_every_episode_without_validation(source, tmp_path):
    stats_path = tmp_path / "all_stats.json"
    cache = tmp_path / "cache.h5"
    with h5py.File(source, "r+") as file:
        file["episodes/clip_name"][:] = np.array(
            [f"a_v{i}" for i in range(4)], dtype=h5py.string_dtype()
        )
    run("compute_stats.py", source, stats_path, "--all-episodes")
    stats = json.loads(stats_path.read_text())
    assert stats["train_episodes"] == [0, 1, 2, 3]
    assert stats == json.loads(json.dumps(compute_statistics(ReferenceWindowDataset(source, range(4)))))

    with h5py.File(cache, "w") as file:
        file.attrs.update(
            complete=True,
            source_id=inspect_source(source)["source_id"],
            encoder_json=json.dumps({"encoder_id": "test-fixture"}),
        )
        file.create_dataset("tokens", shape=(48, 257, 768), dtype="f2", compression="lzf")
        file["episode_index"] = np.repeat(np.arange(4), 12)
        file["frame_index"] = np.tile(np.arange(12), 4)

    args = [
        "--source", source, "--train-only", "--stats", stats_path, "--features", cache,
        "--device", "cpu", "--workers", "0", "--steps", "2", "--batch-size", "2",
        "--save-every", "1", "--eval-every", "1", "--smoke",
    ]
    continuous, resumed = tmp_path / "all_continuous", tmp_path / "all_resumed"
    run("train.py", *args, "--output", continuous)
    run("train.py", *args, "--resume", continuous / "step_00000001", "--output", resumed)
    records = [json.loads(line) for line in (continuous / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2 and all("val_loss" not in record for record in records)
    expected = torch.load(continuous / "step_00000002/model.pt", weights_only=True)
    actual = torch.load(resumed / "step_00000002/model.pt", weights_only=True)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    distributed = tmp_path / "all_distributed"
    run_distributed(*args, "--output", distributed)
    ddp_records = [json.loads(line) for line in (distributed / "metrics.jsonl").read_text().splitlines()]
    assert len(ddp_records) == 2 and all("val_loss" not in record for record in ddp_records)
    metadata = json.loads((distributed / "step_00000002/metadata.json").read_text())
    assert metadata["training_signature"]["world_size"] == 2


def run_distributed(*args):
    environment = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
         str(SCRIPTS / "train.py"), *map(str, args)],
        env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ddp_training_accumulation_and_resume(source, tmp_path):
    # Odd dataset lengths exercise sampler repetition and uneven validation.
    with h5py.File(source, "r+") as f:
        f["episodes/length"][:] = [12, 11, 12, 11]
    split = make_split(source)
    split_path, stats_path, cache = tmp_path / "split.json", tmp_path / "stats.json", tmp_path / "cache.h5"
    split_path.write_text(json.dumps(split))
    dataset = ReferenceWindowDataset(source, split["train_episodes"])
    stats_path.write_text(json.dumps(compute_statistics(dataset)))
    with h5py.File(cache, "w") as f:
        f.attrs.update(
            complete=True, source_id=inspect_source(source)["source_id"],
            encoder_json=json.dumps({"encoder_id": "test-fixture"}),
        )
        f.create_dataset("tokens", shape=(48, 257, 768), dtype="f2", compression="lzf")
        f["episode_index"] = np.repeat(np.arange(4), 12)
        f["frame_index"] = np.tile(np.arange(12), 4)
    args = [
        "--source", source, "--split", split_path, "--stats", stats_path, "--features", cache,
        "--device", "cpu", "--workers", "0", "--steps", "4", "--batch-size", "5",
        "--gradient-accumulation-steps", "2", "--save-every", "1", "--eval-every", "1", "--smoke",
    ]
    continuous, resumed = tmp_path / "ddp", tmp_path / "ddp_resumed"
    run_distributed(*args, "--output", continuous)
    run_distributed(*args, "--output", resumed, "--resume", continuous / "step_00000001")
    expected = torch.load(continuous / "step_00000004/model.pt", weights_only=True)
    actual = torch.load(resumed / "step_00000004/model.pt", weights_only=True)
    for key in expected:
        assert not key.startswith("module.")
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    records = [json.loads(line) for line in (continuous / "metrics.jsonl").read_text().splitlines()]
    resumed_records = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert resumed_records == records[1:]
    assert [record["microbatches"] for record in records] == [2, 1, 2, 1]
    expected_counts = []
    for epoch in range(2):
        orders = []
        for rank in range(2):
            sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42, drop_last=False)
            sampler.set_epoch(epoch)
            orders.append(list(sampler))
        for start, end in [(0, 10), (10, 12)]:
            expected_counts.append(sum(
                int(dataset.numeric_item(index)["reference_mask"].sum())
                for order in orders for index in order[start:end]
            ))
    assert [record["valid_elements"] for record in records] == expected_counts
    validation = ReferenceWindowDataset(source, split["val_episodes"])
    val_count = sum(int(validation.numeric_item(i)["reference_mask"].sum()) for i in range(len(validation)))
    assert all(record["val_valid_elements"] == val_count for record in records)
    state = torch.load(continuous / "step_00000004/training_state.pt", weights_only=True)
    assert state["epoch"] == 2 and state["next_batch"] == 0
    assert len(state["rank_rng_states"]) == 2
    metadata = json.loads((continuous / "step_00000004/metadata.json").read_text())
    assert metadata["training_signature"]["world_size"] == 2
    # A new single-device run can initialize from DDP weights without resuming
    # the incompatible distributed optimizer/sampling protocol.
    run("train.py", *args, "--steps", "1", "--output", tmp_path / "initialized",
        "--init-checkpoint", continuous / "step_00000004")
