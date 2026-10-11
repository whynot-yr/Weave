#!/usr/bin/env python3
"""Run reference-policy evaluation for checkpoint/clip pairs and index successful videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def indices(value: str | None, limit: int) -> list[int]:
    if value is None:
        return list(range(limit))
    selected = []
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))
    if not selected or any(index < 0 or index >= limit for index in selected):
        raise ValueError(f"clip indices must be between 0 and {limit - 1}")
    return sorted(set(selected))


def write_summary(output: Path, rows: list[dict]) -> None:
    columns = ["checkpoint", "clip_id", "repeat", "seed", "status", "success", "reason", "steps", "video", "result_dir"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="WEAVE PPO model_*.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", help="Checkpoint step numbers, comma-separated, e.g. 60000,70000")
    parser.add_argument("--clips", help="Clip indices, e.g. 0-4 or 0,2,4; defaults to all")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per checkpoint/clip pair (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="First run's seed; each repeat adds one")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("/data/g1_hoi_ws/isaacsim/python.sh"))
    parser.add_argument("--pythonexe", type=Path, default=Path("/data/g1_hoi_ws/sim51/bin/python"))
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--replan-steps", type=int, default=1)
    parser.add_argument("--show", action="store_true", help="Show simulator window (default: headless)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching Isaac Sim")
    args = parser.parse_args()
    if args.repeats < 1 or args.seed < 0 or args.seed + args.repeats > 2**32:
        parser.error("--repeats must be positive and all generated seeds must fit uint32")

    checkpoints = sorted(args.reference_root.glob("step_*/config.json"))
    checkpoints = [path.parent for path in checkpoints]
    if args.steps:
        wanted = {int(step) for step in args.steps.split(",")}
        checkpoints = [path for path in checkpoints if int(path.name.removeprefix("step_")) in wanted]
        missing = wanted - {int(path.name.removeprefix("step_")) for path in checkpoints}
        if missing:
            parser.error(f"Missing checkpoint steps: {sorted(missing)}")
    if not checkpoints:
        parser.error("No reference checkpoints found")
    with np.load(args.reference_file, allow_pickle=True) as motion:
        count = len(motion["motion_lengths"])
    try:
        clips = indices(args.clips, count)
    except ValueError as exc:
        parser.error(str(exc))
    for path in (args.reference_file, args.checkpoint, args.dino_repo, args.python, args.pythonexe):
        if not path.exists():
            parser.error(f"Missing path: {path}")

    root = Path(__file__).resolve().parents[2]
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "successful_videos").mkdir(exist_ok=True)
    rows = []
    total = len(checkpoints) * len(clips) * args.repeats
    for number, (checkpoint, clip, repeat) in enumerate(
        ((checkpoint, clip, repeat) for checkpoint in checkpoints for clip in clips
         for repeat in range(args.repeats)), start=1
    ):
        seed = args.seed + repeat
        name = f"{checkpoint.name}_clip{clip}"
        if repeat or seed != 42:
            name += f"_repeat{repeat}_seed{seed}"
        result_dir = args.output / name
        if result_dir.exists() and not (result_dir / "results.json").exists():
            retry = 1
            while (args.output / f"{name}_retry{retry}").exists():
                retry += 1
            result_dir = args.output / f"{name}_retry{retry}"
        command = [
            str(args.python), str(root / "scripts/rsl_rl/play_reference.py"),
            "--device", args.device, "--checkpoint", str(args.checkpoint.resolve()),
            "--reference-checkpoint", str(checkpoint.resolve()),
            "--reference-file", str(args.reference_file.resolve()),
            "--clip-id", str(clip), "--seed", str(seed), "--reference-source", "policy",
            "--dino-repo", str(args.dino_repo.resolve()),
            "--episodes", "1", "--episode-seconds", str(args.episode_seconds),
            "--max-steps", str(args.max_steps), "--replan-steps", str(args.replan_steps),
            "--output", str(result_dir.resolve()), "--video",
        ]
        if not args.show:
            command.append("--headless")
        print(f"[{number}/{total}] {name}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        result_file = result_dir / "results.json"
        if not result_file.exists():
            env = os.environ.copy()
            env.pop("CONDA_PREFIX", None)
            env.pop("CONDA_DEFAULT_ENV", None)
            # Keep the venv executable path: resolving its symlink loses pyvenv.cfg.
            env["PYTHONEXE"] = str(args.pythonexe.absolute())
            result_dir.parent.mkdir(parents=True, exist_ok=True)
            with (args.output / f"{result_dir.name}.log").open("w") as log:
                code = subprocess.run(command, cwd=root, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            status = "complete" if code == 0 and result_file.exists() else f"error:{code}"
        else:
            status = "complete"
        episode = json.loads(result_file.read_text())["episodes"][0] if result_file.exists() else {}
        videos = sorted((result_dir / "videos").glob("*.mp4"))
        video = videos[0] if videos else None
        success = bool(episode.get("success")) and status == "complete" and video is not None
        row = {
            "checkpoint": checkpoint.name, "clip_id": clip, "repeat": repeat, "seed": seed, "status": status,
            "success": success, "reason": episode.get("reason", ""),
            "steps": episode.get("steps", ""),
            "video": str(video.resolve()) if video else "", "result_dir": str(result_dir.resolve()),
        }
        rows.append(row)
        if success:
            link_dir = args.output / "successful_videos"
            link_dir.mkdir(exist_ok=True)
            link = link_dir / f"{name}.mp4"
            if not link.exists() and not link.is_symlink():
                link.symlink_to(os.path.relpath(video, link_dir))
        write_summary(args.output, rows)
        print(f"  {status}: {episode.get('reason', 'no result')}" + (" ✓" if success else ""), flush=True)
    if not args.dry_run:
        print(f"Summary: {args.output / 'summary.csv'}", flush=True)
        print(f"Successful videos: {args.output / 'successful_videos'}", flush=True)
        if any(row["status"] != "complete" for row in rows):
            sys.exit(1)


if __name__ == "__main__":
    main()
