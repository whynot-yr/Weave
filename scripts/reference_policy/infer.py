"""Generate a physical-unit reference chunk from live-style RGB or cached tokens."""

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import h5py
import numpy as np
import torch
from weave_policy.data.reference_codec import split_reference
from weave_policy.predictor import ReferencePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--feature-cache")
    inputs.add_argument("--rgb", help="uint8 RGB .npy [224,224,3] or [B,224,224,3]")
    inputs.add_argument("--source", help="Recorded rollout HDF5; read one RGB frame")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dino-repo")
    parser.add_argument("--dino-weights")
    parser.add_argument("--dino-ref", default="main")
    args = parser.parse_args()
    if min(args.row, args.episode, args.frame) < 0:
        parser.error("Row, episode and frame must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    predictor = ReferencePredictor(args.checkpoint, args.device)
    if args.feature_cache:
        with h5py.File(args.feature_cache, "r") as file:
            if not file.attrs.get("complete", False):
                raise ValueError("Incomplete cache")
            metadata = json.loads(file.attrs["encoder_json"])
            tokens = torch.from_numpy(file["tokens"][args.row].astype(np.float32)).unsqueeze(0)
        reference = predictor.predict_tokens(tokens, metadata["encoder_id"], seed=args.seed, num_steps=args.num_steps)
    else:
        predictor.load_encoder(args.dino_repo, args.dino_weights, args.dino_ref)
        if args.source:
            with h5py.File(args.source, "r") as file:
                if (
                    args.frame >= file["episodes/length"][args.episode]
                    or not file["frame/valid"][args.episode, args.frame]
                ):
                    raise ValueError("Requested frame is invalid")
                rgb = file["camera/head/rgb"][args.episode, args.frame]
        else:
            rgb = np.load(args.rgb, allow_pickle=False)
        rgb = torch.from_numpy(rgb)
        rgb = rgb.unsqueeze(0) if rgb.ndim == 3 else rgb
        reference = predictor.predict_rgb(rgb, seed=args.seed, num_steps=args.num_steps)
    parts = split_reference(reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez(
            stream,
            reference=reference.numpy(),
            **{key: value.numpy() for key, value in parts.items()},
            reference_offsets=np.array(predictor.policy.config.reference_offsets),
            fps=predictor.policy.config.fps,
        )
    print(f"saved {args.output}: reference={tuple(reference.shape)} (physical units, pelvis-local)")


if __name__ == "__main__":
    main()
