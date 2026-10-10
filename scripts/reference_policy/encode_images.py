"""Stream every valid image through frozen DINOv2 into a separate token HDF5."""

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import h5py
import numpy as np
import torch
from weave_policy.data.reference_dataset import inspect_source
from weave_policy.models.dinov2_encoder import DinoV2Encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--max-frames", type=int, help="Extract a prefix only; not a full training cache")
    parser.add_argument("--dino-repo")
    parser.add_argument("--dino-weights")
    parser.add_argument("--dino-ref", default="main")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("Batch size and max-frames must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    info = inspect_source(args.source)
    episodes = sorted(args.episodes) if args.episodes is not None else list(range(len(info["episodes"])))
    if len(set(episodes)) != len(episodes) or not set(episodes) <= set(range(len(info["episodes"]))):
        raise ValueError("Invalid episode selection")
    frames = [(episode, step) for episode in episodes for step in range(info["episodes"][episode]["length"])]
    frames = frames[: args.max_frames] if args.max_frames is not None else frames
    if not frames:
        raise ValueError("No valid frames")
    encoder = DinoV2Encoder.from_pretrained(args.dino_repo, args.dino_weights, args.dino_ref).to(args.device)
    metadata = encoder.metadata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(info["source"], "r") as source, h5py.File(args.output, "x") as output:
        output.attrs.update(complete=False, source_id=info["source_id"], encoder_json=json.dumps(metadata))
        output.create_dataset("episode_index", data=np.array([frame[0] for frame in frames], dtype=np.int32))
        output.create_dataset("frame_index", data=np.array([frame[1] for frame in frames], dtype=np.int32))
        tokens = output.create_dataset(
            "tokens", shape=(len(frames), 257, 768), dtype="float16", chunks=(1, 257, 768), compression="lzf"
        )
        with torch.no_grad():
            for start in range(0, len(frames), args.batch_size):
                batch = frames[start : start + args.batch_size]
                if any(not source["frame/valid"][episode, step] for episode, step in batch):
                    raise ValueError("Missing frame in source")
                rgb = np.stack([source["camera/head/rgb"][episode, step] for episode, step in batch])
                value = encoder(torch.from_numpy(rgb)).float().cpu().numpy().astype(np.float16)
                if not np.isfinite(value).all():
                    raise FloatingPointError("Nonfinite float16 feature cache")
                tokens[start : start + len(batch)] = value
                if start == 0 or start // args.batch_size % 100 == 0:
                    print(f"encoded {start + len(batch)}/{len(frames)} frames", flush=True)
        output.attrs["complete"] = True
    print(f"saved {args.output}; encoder={metadata['encoder_id']}")


if __name__ == "__main__":
    main()
