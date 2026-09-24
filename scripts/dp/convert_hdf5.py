"""Convert recorded WEAVE episodes without touching the source HDF5."""

import argparse

from weave_data.hdf5_converter import convert_hdf5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--repo-id", default="weave/local")
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--images", action="store_true", help="Store images instead of compressed video")
    args = parser.parse_args()
    convert_hdf5(args.source, args.output, args.repo_id, args.episodes, args.success_only, not args.images)


if __name__ == "__main__":
    main()
