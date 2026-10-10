"""Split recorded episodes by object and original motion family."""

import argparse

from _common import write_new_json
from weave_policy.data.split import make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-only", action="store_true")
    args = parser.parse_args()
    split = make_split(args.source, args.val_ratio, args.seed, args.success_only)
    write_new_json(args.output, split)
    print(f"train episodes={len(split['train_episodes'])}, val episodes={len(split['val_episodes'])}")


if __name__ == "__main__":
    main()
