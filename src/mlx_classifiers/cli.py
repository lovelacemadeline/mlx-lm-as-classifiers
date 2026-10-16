"""``mlx-classifiers`` console entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from . import bench

    parser = argparse.ArgumentParser(prog="mlx-classifiers")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench", help="decisions/s sweep over state length (story 2.4)")
    bench.add_arguments(b)
    b.set_defaults(func=bench.main)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
