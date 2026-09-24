"""Compute normalization statistics on the specified TRAIN episodes only."""

import argparse
import json
from pathlib import Path

from weave_data.dp_dataset import DPWindowDataset
from weave_data.dp_stats import compute_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument(
        "--episodes", nargs="+", type=int, required=True, help="LeRobot episode indices, not original HDF5 indices"
    )
    parser.add_argument("--horizon", type=int, default=40)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dataset = DPWindowDataset(args.dataset, args.episodes, args.horizon)
    payload = compute_stats(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Saved stats for {len(dataset)} windows to {output}")


if __name__ == "__main__":
    main()
