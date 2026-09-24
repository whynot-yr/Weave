import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from weave_data.dp_codec import PROTOCOL
from weave_data.dp_split import load_split, make_split, motion_family, validate_split


def protocol(n=33):
    episodes = []
    for i in range(n):
        for v in range(3):
            episodes.append(
                {
                    "episode_index": len(episodes),
                    "clip_name": f"sub17_lamp_{i:03}_v{v:02}",
                    "length": 10 + i,
                    "object_name": "lamp",
                }
            )
    return {"protocol": PROTOCOL, "complete": True, "dataset_id": "test", "episodes": episodes}


def test_reproducible_family_split():
    p = protocol()
    split = make_split(p)
    assert split == make_split(p)
    reversed_p = copy.deepcopy(p)
    reversed_p["episodes"].reverse()
    assert split == make_split(reversed_p)
    assert split["train_summary"]["groups"] == 26
    assert split["val_summary"]["groups"] == 7
    assert len(split["train_episodes"]) + len(split["val_episodes"]) == 99
    assert split != make_split(p, seed=43)
    assert motion_family("clip_v00_middle") == "clip_v00_middle"
    assert motion_family("clip_v123") == "clip"


def test_stratified_split():
    p = protocol(4)
    for e in p["episodes"][6:]:
        e["object_name"] = "chair"
    split = make_split(p, by_object=True)
    for part in ("train", "val"):
        assert {e["object_name"] for e in p["episodes"] if e["episode_index"] in split[f"{part}_episodes"]} == {
            "lamp",
            "chair",
        }
    del p["episodes"][0]["object_name"]
    with pytest.raises(ValueError, match="object_name"):
        make_split(p, by_object=True)


@pytest.mark.parametrize("ratio", [0, 1, -0.1, float("nan")])
def test_bad_ratio(ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        make_split(protocol(), val_ratio=ratio)


def test_single_family_rejected():
    with pytest.raises(ValueError, match="fewer than two"):
        make_split(protocol(1))


def test_validation_rejects_tampering():
    p = protocol(4)
    split = make_split(p)
    wrong = copy.deepcopy(split)
    wrong["dataset_id"] = "another"
    with pytest.raises(ValueError, match="dataset_id"):
        validate_split(wrong, p)
    wrong = copy.deepcopy(split)
    wrong["train_episodes"].append(wrong["train_episodes"][0])
    with pytest.raises(ValueError, match="unique"):
        validate_split(wrong, p)
    wrong = copy.deepcopy(split)
    wrong["train_episodes"].pop()
    with pytest.raises(ValueError, match="cover"):
        validate_split(wrong, p)
    wrong = copy.deepcopy(split)
    moved = wrong["val_episodes"].pop()
    wrong["train_episodes"].append(moved)
    wrong["train_groups"] = sorted(set(wrong["train_groups"] + [motion_family(p["episodes"][moved]["clip_name"])]))
    with pytest.raises(ValueError, match="families overlap"):
        validate_split(wrong, p)


def test_cli_and_load(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/weave_protocol.json").write_text(json.dumps(protocol()))
    output = tmp_path / "split.json"
    script = Path(__file__).resolve().parents[3] / "scripts/dp/split_dataset.py"
    cmd = [sys.executable, str(script), str(root), str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert load_split(output, root) == make_split(protocol())
    before = output.read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert output.read_bytes() == before
    train_script = script.with_name("train.py")
    parse_args = runpy.run_path(str(train_script))["parse_args"]
    argv = [
        str(train_script),
        "--dataset",
        str(root),
        "--stats",
        "stats.json",
        "--output",
        "train",
        "--split",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    assert args.train_episodes == load_split(output, root)["train_episodes"]
    assert args.eval_episodes == load_split(output, root)["val_episodes"]
    monkeypatch.setattr(sys, "argv", argv + ["--eval-episodes", "0"])
    with pytest.raises(SystemExit):
        parse_args()
