"""Image decisions/s, uncached vs delta-cached transcripts (story 6.3).

``python -m mlx_classifiers.vision_bench [--model REPO] [--int8]`` over the
committed gridworld frames (``battery/gridworld.jsonl``, story 6.5): one
``choice`` question (the gridworld's action question, six options) per
decision, timed end to end through :func:`~mlx_classifiers.decide.decide`
(compile, PNG decode, encode, prefill, readout), as a server pays it. The
declared prefix is cached once outside the timed loop, as a server keeps it.

Cells:

- ``single new``: state ``[colour key, frame]``, a different frame each
  decision, delta cache off: every decision encodes its image.
- ``single same``: the same frame every decision, delta cache off: the image
  is memoised, so this is the language-model cost alone.
- ``transcript``: a gridworld transcript ``[frame 0, "Answer: a0", frame 1,
  "Answer: a1", ..., frame k]`` grown one (answer, frame) per decision over
  six consecutive ticks of one episode; only the newest frame is new. Run with
  the delta cache on (``delta``) and off (``uncached``); old frames' soft
  tokens are memoised in both, so the difference is KV reuse alone. One row
  per depth, the median over the episodes.

``encoder ms`` is the vision tower time inside the decision (0 when memoised).
Writes ``bench_runs/vision-bench-<timestamp>.{md,json}`` with power state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .decide import decide
from .schema import SystemOneRequest

__all__ = ["VisionRow", "load_frames", "main", "run_vision_bench"]

BATTERY = Path(__file__).resolve().parents[2] / "battery" / "gridworld.jsonl"


@dataclass
class VisionRow:
    cell: str
    depth: int  # frames in the state
    prompt_tokens: int
    prefilled_tokens: int
    cached_tokens: int
    p50_ms: float
    encoder_ms: float
    decisions_per_s: float
    n: int


def load_frames(path: Path = BATTERY) -> dict[str, Any]:
    """``{"prefix", "question", "key", "episodes": [[(image block, gold), ...], ...]}``
    from the battery's ``gridworld-image`` slice, episodes in tick order."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    image = [r for r in rows if r["tags"]["source"] == "gridworld-image"]
    episodes: dict[str, list] = {}
    for r in image:
        seed = r["id"].split("-t")[0]
        episodes.setdefault(seed, []).append((r["state"][1], r["gold"]["action"]))
    first = image[0]
    return {
        "prefix": first["prefix"],
        "question": first["questions"]["action"],
        "key": first["state"][0],
        "episodes": list(episodes.values()),
    }


def _request(data: dict[str, Any], state: list[Any], delta: bool) -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {
            "prefix": data["prefix"],
            "state": state,
            "questions": {"action": data["question"]},
            "options": {"delta_cache": delta},
        }
    )


def _timed(runner: Any, req: SystemOneRequest) -> tuple[float, float, Any]:
    before = len(runner.vision_encode_ms)
    t = time.perf_counter()
    out = decide(req, runner)
    dt = (time.perf_counter() - t) * 1e3
    enc = sum(runner.vision_encode_ms[before:])
    return dt, enc, out


def _row(cell, depth, samples) -> VisionRow:
    ms = statistics.median(s[0] for s in samples)
    out = samples[-1][2]
    prompt = out.usage.input_tokens + (out.usage.cached_tokens or 0)
    return VisionRow(
        cell=cell,
        depth=depth,
        prompt_tokens=prompt,
        prefilled_tokens=out.usage.input_tokens,
        cached_tokens=out.usage.cached_tokens or 0,
        p50_ms=ms,
        encoder_ms=statistics.median(s[1] for s in samples),
        decisions_per_s=1e3 / ms,
        n=len(samples),
    )


def run_vision_bench(runner: Any, data: dict[str, Any], warmup: int = 2) -> list[VisionRow]:
    from .prompt_cache import PromptCacheStore

    tower = runner.load_vision()
    frames = [f for ep in data["episodes"] for f, _ in ep]
    rows: list[VisionRow] = []

    # single image, a new one each decision
    for f in frames[:warmup]:
        _timed(runner, _request(data, [data["key"], f], False))
    tower.clear()
    samples = [_timed(runner, _request(data, [data["key"], f], False)) for f in frames]
    rows.append(_row("single new", 1, samples))

    # single image, the same one each decision (memoised)
    same = _request(data, [data["key"], frames[0]], False)
    _timed(runner, same)
    rows.append(_row("single same", 1, [_timed(runner, same) for _ in range(len(frames))]))

    # transcripts: only the newest frame is new
    episodes = data["episodes"]
    for cell, delta in (("transcript uncached", False), ("transcript delta", True)):
        by_depth: dict[int, list] = {}
        for i, ep in enumerate(episodes):
            runner.prompt_cache = PromptCacheStore()
            tower.clear()
            state: list[Any] = []
            previous: str | None = None
            for depth, (frame, gold) in enumerate(ep, start=1):
                if previous is not None:
                    state.append(f"Answer: {previous}")
                state.append(frame)
                sample = _timed(runner, _request(data, list(state), delta))
                if i > 0:  # episode 0 warms up
                    by_depth.setdefault(depth, []).append(sample)
                previous = gold
        rows.extend(_row(cell, d, s) for d, s in sorted(by_depth.items()))
    return rows


def render_markdown(env: dict[str, Any], rows: list[VisionRow]) -> str:
    head = [
        "# Image decisions/s benchmark",
        "",
        (
            f"- model: `{env['model']}`  git: `{env['git_rev']}`  int8: {env.get('int8')}"
            f"  vision_fast: {env.get('vision_fast')}"
            f"  vision_soft_tokens: {env.get('vision_soft_tokens', 280)}"
        ),
        f"- machine: {env['machine']} ({env['platform']})",
        f"- power: `{env['lowpowermode']}`; {env['power_source']}",
        f"- versions: {env['versions']}  timestamp: {env['timestamp']}",
        (
            f"- power after: `{env.get('lowpowermode_after')}`; peak MLX memory "
            f"{env.get('peak_memory_gb')} GB; vm swapouts during the run "
            f"{(env.get('swapouts_after') or 0) - (env.get('swapouts_before') or 0)}"
        ),
        "",
        (
            "| cell | frames | prompt tok | prefilled tok | cached tok | p50 ms | encoder ms "
            "| decisions/s | n |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    body = [
        f"| {r.cell} | {r.depth} | {r.prompt_tokens} | {r.prefilled_tokens} | {r.cached_tokens} "
        f"| {r.p50_ms:.0f} | {r.encoder_ms:.0f} | {r.decisions_per_s:.2f} | {r.n} |"
        for r in rows
    ]
    return "\n".join(head + body) + "\n"


def _swapouts() -> int | None:
    """macOS ``vm_stat`` swapouts so far (memory pressure during the run shows as growth)."""
    import subprocess

    try:
        out = subprocess.run(["vm_stat"], check=False, capture_output=True, text=True).stdout
    except OSError:
        return None
    line = next((ln for ln in out.splitlines() if ln.startswith("Swapouts")), "")
    digits = "".join(ch for ch in line if ch.isdigit())
    return int(digits) if digits else None


def main(argv: list[str] | None = None) -> int:
    from .bench import environment
    from .mlx_runner import DEFAULT_MODEL, MLXRunner

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--int8", action="store_true")
    p.add_argument(
        "--vision-slow", action="store_true", help="the vendored encode path (story 6.6 reference)"
    )
    p.add_argument(
        "--vision-tokens",
        type=int,
        choices=(70, 140, 280),
        default=None,
        help="soft-token budget per image (story 6.7; default: the checkpoint's 280)",
    )
    p.add_argument("--battery", type=Path, default=BATTERY)
    p.add_argument("--out", type=Path, default=Path("bench_runs"))
    args = p.parse_args(argv)
    runner = MLXRunner.load(
        args.model,
        int8=args.int8,
        vision=True,
        vision_fast=not args.vision_slow,
        vision_soft_tokens=args.vision_tokens,
    )
    env = environment(args.model) | {
        "int8": args.int8,
        "vision_fast": not args.vision_slow,
        "vision_soft_tokens": runner.image_tokens.max_soft_tokens,
    }
    import mlx.core as mx

    env["swapouts_before"] = _swapouts()
    rows = run_vision_bench(runner, load_frames(args.battery))
    env["lowpowermode_after"] = environment(args.model)["lowpowermode"]
    env["swapouts_after"] = _swapouts()
    env["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md = args.out / f"vision-bench-{stamp}.md"
    md.write_text(render_markdown(env, rows))
    (args.out / f"vision-bench-{stamp}.json").write_text(
        json.dumps({"env": env, "rows": [asdict(r) for r in rows]}, indent=2)
    )
    print(md.read_text())
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
