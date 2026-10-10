"""Compute training-window statistics after pelvis coordinate conversion."""

import argparse

from _common import read_config, write_new_json
from weave_policy.data.normalization import compute_statistics
from weave_policy.data.reference_dataset import ReferenceWindowDataset
from weave_policy.data.split import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--split", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = read_config(args.config)
    split = load_split(args.split, args.source)
    dataset = ReferenceWindowDataset(args.source, split["train_episodes"], config.reference_offsets)
    stats = compute_statistics(dataset)
    write_new_json(args.output, stats)
    print(f"saved {stats['valid_target_rows']} valid target rows (padding excluded)")


if __name__ == "__main__":
    main()
