"""Split converted LeRobot episodes by original motion, never by frame/window."""

import argparse
import json
from pathlib import Path

from weave_data.dp_split import make_split, read_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--by-object", action="store_true", help="Split each object separately; requires object_name metadata"
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    split = make_split(read_protocol(args.dataset), args.val_ratio, args.seed, args.by_object)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(split, f, indent=2)
        f.write("\n")
    for name in ("train", "val"):
        print(f"{name}: {split[f'{name}_summary']}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
