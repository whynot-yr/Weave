"""Exercise split/stat/train/resume/infer entry points as separate processes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
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


def test_training_resume_and_inference(source, tmp_path):
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
        "cpu",
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
        assert f["reference"].shape == (1, 10, 125)
        assert f["ref_robot"].shape == (1, 10, 62)
        assert f["ref_object"].shape == (1, 10, 63)
        assert np.isfinite(f["reference"]).all()
