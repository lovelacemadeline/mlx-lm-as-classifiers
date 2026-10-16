"""``mlx-classifiers`` console entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from . import battery, bench, fitting, fixtures

    parser = argparse.ArgumentParser(prog="mlx-classifiers")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench", help="decisions/s sweep over state length (story 2.4)")
    bench.add_arguments(b)
    b.set_defaults(func=bench.main)
    f = sub.add_parser("fixtures", help="golden fixtures: generate and check (Epic 4)")
    fixtures.add_arguments(f)
    sv = sub.add_parser("serve", help="/v1/systemone + /v1/chat/completions on one model (3.1)")
    _add_server_arguments(sv)
    c = sub.add_parser("calibrate", help="fit temperatures from labelled decisions (3.2)")
    fitting.add_arguments(c)
    c.set_defaults(func=fitting.main)
    bt = sub.add_parser("battery", help="static decision battery: accuracy, ECE, bias (5.2)")
    battery.add_arguments(bt)
    args = parser.parse_args(argv)
    return args.func(args)


def _add_server_arguments(p: argparse.ArgumentParser) -> None:
    def run(args: argparse.Namespace) -> int:
        from . import server  # needs the ``server`` extra

        return server.main(args)

    from .mlx_runner import DEFAULT_MODEL

    p.add_argument("--model", default=DEFAULT_MODEL, help="mlx-lm repo or path to load")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--prefix-cache-size", type=int, default=16, help="prefix KV caches kept (LRU)")
    p.add_argument(
        "--max-tokens-ceiling",
        type=int,
        default=8192,
        help="largest max_tokens a chat request may ask for (above it: 422)",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration config (default: $MLX_CLASSIFIERS_CALIBRATION or ./calibration.json)",
    )
    p.add_argument(
        "--log", type=Path, default=None, help="append every served decision to this JSONL file"
    )
    p.set_defaults(func=run)


if __name__ == "__main__":
    sys.exit(main())
