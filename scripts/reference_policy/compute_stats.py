"""Compute training-window statistics after pelvis coordinate conversion."""

import argparse

from _common import read_config, write_new_json
from weave_policy.data.normalization import compute_statistics
from weave_policy.data.reference_dataset import ReferenceWindowDataset, inspect_source
from weave_policy.data.split import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--split")
    selection.add_argument("--all-episodes", action="store_true", help="Compute statistics from every recorded episode")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = read_config(args.config)
    episodes = (
        list(range(len(inspect_source(args.source)["episodes"])))
        if args.all_episodes else load_split(args.split, args.source)["train_episodes"]
    )
    dataset = ReferenceWindowDataset(args.source, episodes, config.reference_offsets)
    try:
        stats = compute_statistics(dataset)
    finally:
        dataset.close()
    write_new_json(args.output, stats)
    print(f"saved {stats['valid_target_rows']} valid target rows (padding excluded)")


if __name__ == "__main__":
    main()
