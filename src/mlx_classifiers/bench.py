"""Decisions-per-second benchmark (story 2.4).

Sweeps state length and prefix caching for one ``choice`` question and reports,
per cell: decisions/s, p50/p95 latency, and the prefill tok/s it implies
(tokens actually prefilled per decision / median latency).

A *decision* here is what a server pays per request once the prompt is
compiled: ``Runner.prefill`` over the prompt (from the shared prefix cache in
the cached cells; the cache is built once, outside the timed loop, as a server
would keep it) plus the masked-softmax answer. Prompt compilation and label
binding are done once per cell and not timed.

Writes ``bench_runs/bench-<timestamp>.md`` and ``.json`` with the model id,
git revision, machine and power state (macOS ``pmset``; recorded, never
changed). The INT8 column is the runner's ``int8`` flag (story 2.3); run with
``--int8`` to load the W8A8 kernels.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .answers import build_answer
from .prompt import compile_question
from .readout import bind_options
from .schema import SystemOneRequest

__all__ = ["BenchRow", "environment", "filler_state", "main", "render_markdown", "run_bench"]

DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048)
PREFIX = (
    "You are the support triage router for a payments company. Read the ticket and "
    "answer the question about it. Route billing problems (payments, refunds, invoices, "
    "payouts) to billing, product defects and outages to technical, and pricing or new "
    "contracts to sales. Be precise; answer with a single option."
)
FILLER = (
    "The merchant reports that scheduled payouts did not arrive and that the dashboard "
    "shows the transfers as pending while their bank sees nothing. "
)
QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {"billing": "Payments, payouts", "technical": "Bugs, outages", "sales": None},
}


def filler_state(tokenize, n_tokens: int) -> tuple[str, int]:
    """Repeated filler text with the most words whose token count is <= ``n_tokens``.
    Returns ``(text, token_count)``."""
    words = FILLER.split()

    def text(k: int) -> str:
        return " ".join(words[i % len(words)] for i in range(k))

    lo, hi = 1, max(2, n_tokens * 2)
    while lo < hi:  # largest k with len(tokenize(text(k))) <= n_tokens
        mid = (lo + hi + 1) // 2
        if len(tokenize(text(mid))) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text(lo), len(tokenize(text(lo)))


@dataclass
class BenchRow:
    state_tokens: int
    prompt_tokens: int
    prefix_cached: bool
    int8: bool
    prefilled_tokens: int
    reps: int
    p50_ms: float
    p95_ms: float
    decisions_per_s: float
    prefill_tok_per_s: float


def _percentile(xs: Sequence[float], q: float) -> float:
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _decision(runner: Any, prompt: Any, binding: Any, cache: Any) -> int:
    """One timed decision: prefill (from ``cache``) + masked-softmax answer."""
    r = runner.prefill(prompt.tokens, cache, [prompt.readout_position])
    build_answer(prompt, binding, r.at(prompt.readout_position))
    return r.input_tokens


def run_bench(
    runner: Any,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    reps: int = 20,
    warmup: int = 2,
    cached: Sequence[bool] = (False, True),
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for n in lengths:
        state, n_state = filler_state(runner.tokenize, n)
        req = SystemOneRequest.model_validate(
            {"prefix": PREFIX, "state": state, "questions": {"q": QUESTION}}
        )
        prompt = compile_question(req, "q", template=runner.chat_template, tokenize=runner.tokenize)
        binding = bind_options(prompt, runner.tokenize)
        for use_cache in cached:
            cache = runner.cache_prefix(prompt.prefix_tokens) if use_cache else None

            for _ in range(warmup):
                _decision(runner, prompt, binding, cache)
            times, prefilled = [], 0
            for _ in range(reps):
                t = time.perf_counter()
                prefilled = _decision(runner, prompt, binding, cache)
                times.append(time.perf_counter() - t)
            p50 = _percentile(times, 0.5)
            rows.append(
                BenchRow(
                    state_tokens=n_state,
                    prompt_tokens=len(prompt.tokens),
                    prefix_cached=use_cache,
                    int8=bool(getattr(runner, "int8", False)),
                    prefilled_tokens=prefilled,
                    reps=reps,
                    p50_ms=p50 * 1e3,
                    p95_ms=_percentile(times, 0.95) * 1e3,
                    decisions_per_s=1.0 / p50,
                    prefill_tok_per_s=prefilled / p50,
                )
            )
    return rows


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment(model_id: str) -> dict[str, Any]:
    """Model, git revision, machine and power state (macOS pmset; read-only)."""
    pmset = _run(["pmset", "-g"])
    low_power = next((ln.strip() for ln in pmset.splitlines() if "lowpowermode" in ln), "")
    batt = _run(["pmset", "-g", "batt"]).splitlines()
    rev = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain", "--untracked-files=no"]))
    versions = {}
    for mod in ("mlx", "mlx_lm"):
        try:
            versions[mod] = __import__(mod).__version__ if mod != "mlx" else _mlx_version()
        except Exception:  # noqa: BLE001 - optional dependency
            versions[mod] = None
    return {
        "model": model_id,
        "git_rev": rev + ("-dirty" if dirty else ""),
        "machine": _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "platform": platform.platform(),
        "lowpowermode": low_power,
        "power_source": batt[0].strip() if batt else "",
        "versions": versions,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _mlx_version() -> str:
    import mlx.core as mx

    return mx.__version__


def render_markdown(env: dict[str, Any], rows: Sequence[BenchRow]) -> str:
    head = [
        "# Decisions/s benchmark",
        "",
        f"- model: `{env['model']}`  git: `{env['git_rev']}`",
        f"- machine: {env['machine']} ({env['platform']})",
        f"- power: `{env['lowpowermode']}`; {env['power_source']}",
        f"- versions: {env['versions']}  timestamp: {env['timestamp']}",
        "",
        (
            "| state tok | prompt tok | prefix cached | int8 | prefilled tok | p50 ms | p95 ms "
            "| decisions/s | prefill tok/s |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    body = [
        f"| {r.state_tokens} | {r.prompt_tokens} | {'yes' if r.prefix_cached else 'no'} "
        f"| {'on' if r.int8 else 'off'} | {r.prefilled_tokens} | {r.p50_ms:.1f} "
        f"| {r.p95_ms:.1f} | {r.decisions_per_s:.1f} | {r.prefill_tok_per_s:.0f} |"
        for r in rows
    ]
    return "\n".join(head + body) + "\n"


def write_results(
    env: dict[str, Any], rows: Sequence[BenchRow], out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md, js = out_dir / f"bench-{stamp}.md", out_dir / f"bench-{stamp}.json"
    md.write_text(render_markdown(env, rows))
    js.write_text(json.dumps({"env": env, "rows": [asdict(r) for r in rows]}, indent=2))
    return md, js


def add_arguments(p: argparse.ArgumentParser) -> None:
    from .mlx_runner import DEFAULT_MODEL

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("bench_runs"))
    p.add_argument("--int8", action="store_true", help="load with the W8A8 INT8 kernels (M5)")


def main(args: argparse.Namespace) -> int:
    from .mlx_runner import MLXRunner

    runner = MLXRunner.load(args.model, int8=args.int8)
    env = environment(args.model)
    rows = run_bench(runner, args.lengths, reps=args.reps, warmup=args.warmup)
    md, js = write_results(env, rows, args.out)
    print(md.read_text())
    print(f"wrote {md} and {js}")
    return 0
