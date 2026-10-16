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

Delta-tail cells (story 3.4, ``--delta-tail N``): a state of ``L`` tokens whose
last ``N`` change between consecutive decisions (a fresh filler tail per
decision, differing from the previous one at its first token), timed through
the runner's delta prefix cache, as a sequential controller's ticks would be.
The store is cleared at the start of each cell; the warmup decisions fill it.
Every other cell runs with the delta cache off, so it measures what it always
did.
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

__all__ = [
    "BenchRow",
    "delta_states",
    "environment",
    "filler_state",
    "main",
    "render_markdown",
    "run_bench",
]

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


def _words(start: int, k: int) -> str:
    words = FILLER.split()
    return " ".join(words[(start + i) % len(words)] for i in range(k))


def delta_states(tokenize, n_tokens: int, tail: int, count: int) -> list[str]:
    """``count`` states of about ``n_tokens`` tokens sharing everything but the last
    ``tail`` or so: a fixed filler head, then a filler tail starting at word ``k``
    for state ``k`` (adjacent filler words differ, so consecutive tails differ
    from their first token)."""
    head, _ = filler_state(tokenize, max(1, n_tokens - tail))
    out = []
    for k in range(count):
        lo, hi = 1, max(2, tail * 2)
        while lo < hi:  # most words whose tail stays within ``tail`` tokens
            mid = (lo + hi + 1) // 2
            if len(tokenize(" " + _words(k, mid))) <= tail:
                lo = mid
            else:
                hi = mid - 1
        out.append(head + " " + _words(k, lo))
    return out


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
    #: Story 3.4: tokens changing per decision (None: a fixed state, delta cache off).
    delta_tail: int | None = None
    #: Median tokens served from cache per decision (declared prefix or delta hit).
    cached_tokens: int = 0


def _percentile(xs: Sequence[float], q: float) -> float:
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _decision(
    runner: Any, prompt: Any, binding: Any, cache: Any, delta: bool = False
) -> tuple[int, int]:
    """One timed decision: prefill (from ``cache``) + masked-softmax answer.
    Returns ``(prefilled, cached)`` tokens."""
    kw = {"delta": delta} if hasattr(runner, "prompt_cache") else {}
    r = runner.prefill(prompt.tokens, cache, [prompt.readout_position], **kw)
    build_answer(prompt, binding, r.at(prompt.readout_position))
    return r.input_tokens, r.cached_tokens


def _row(runner: Any, n_state: int, n_prompt: int, use_cache: bool, times, prefilled, cached_t):
    p50 = _percentile(times, 0.5)
    return BenchRow(
        state_tokens=n_state,
        prompt_tokens=n_prompt,
        prefix_cached=use_cache,
        int8=bool(getattr(runner, "int8", False)),
        prefilled_tokens=prefilled,
        reps=len(times),
        p50_ms=p50 * 1e3,
        p95_ms=_percentile(times, 0.95) * 1e3,
        decisions_per_s=1.0 / p50,
        prefill_tok_per_s=prefilled / p50,
        cached_tokens=cached_t,
    )


def _delta_cell(runner: Any, n: int, tail: int, reps: int, warmup: int) -> BenchRow:
    """Consecutive decisions over states whose last ``tail`` tokens change."""
    states = delta_states(runner.tokenize, n, tail, warmup + reps)
    prompts = []
    for st in states:
        req = SystemOneRequest.model_validate(
            {"prefix": PREFIX, "state": st, "questions": {"q": QUESTION}}
        )
        p = compile_question(req, "q", template=runner.chat_template, tokenize=runner.tokenize)
        prompts.append((p, bind_options(p, runner.tokenize)))
    cache = runner.cache_prefix(prompts[0][0].prefix_tokens, delta=False)
    if getattr(runner, "prompt_cache", None) is not None:
        runner.prompt_cache.clear()
    for p, b in prompts[:warmup]:
        _decision(runner, p, b, cache, delta=True)
    times, done = [], []
    for p, b in prompts[warmup:]:
        t = time.perf_counter()
        done.append(_decision(runner, p, b, cache, delta=True))
        times.append(time.perf_counter() - t)
    prefilled = int(_percentile([d[0] for d in done], 0.5))
    cached_t = int(_percentile([d[1] for d in done], 0.5))
    n_state = len(runner.tokenize(states[-1]))
    row = _row(runner, n_state, len(prompts[-1][0].tokens), True, times, prefilled, cached_t)
    row.delta_tail = tail
    return row


def run_bench(
    runner: Any,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    reps: int = 20,
    warmup: int = 2,
    cached: Sequence[bool] = (False, True),
    delta_tails: Sequence[int] = (),
) -> list[BenchRow]:
    """Per length: one cell per ``cached`` flag (delta cache off), then one
    delta-tail cell per ``delta_tails`` entry (story 3.4)."""
    if delta_tails and not hasattr(runner, "prompt_cache"):
        raise ValueError("delta-tail cells need a runner with a prompt_cache")
    rows: list[BenchRow] = []
    for n in lengths:
        state, n_state = filler_state(runner.tokenize, n)
        req = SystemOneRequest.model_validate(
            {"prefix": PREFIX, "state": state, "questions": {"q": QUESTION}}
        )
        prompt = compile_question(req, "q", template=runner.chat_template, tokenize=runner.tokenize)
        binding = bind_options(prompt, runner.tokenize)
        for use_cache in cached:
            kw = {"delta": False} if hasattr(runner, "prompt_cache") else {}
            cache = runner.cache_prefix(prompt.prefix_tokens, **kw) if use_cache else None

            for _ in range(warmup):
                _decision(runner, prompt, binding, cache)
            times, prefilled, cached_t = [], 0, 0
            for _ in range(reps):
                t = time.perf_counter()
                prefilled, cached_t = _decision(runner, prompt, binding, cache)
                times.append(time.perf_counter() - t)
            rows.append(
                _row(runner, n_state, len(prompt.tokens), use_cache, times, prefilled, cached_t)
            )
        for tail in delta_tails:
            rows.append(_delta_cell(runner, n, tail, reps, warmup))
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
            "| state tok | prompt tok | prefix cached | delta tail | int8 | prefilled tok "
            "| cached tok | p50 ms | p95 ms | decisions/s | prefill tok/s |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    body = [
        f"| {r.state_tokens} | {r.prompt_tokens} | {'yes' if r.prefix_cached else 'no'} "
        f"| {r.delta_tail if r.delta_tail is not None else '-'} "
        f"| {'on' if r.int8 else 'off'} | {r.prefilled_tokens} | {r.cached_tokens} "
        f"| {r.p50_ms:.1f} | {r.p95_ms:.1f} | {r.decisions_per_s:.1f} "
        f"| {r.prefill_tok_per_s:.0f} |"
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
    p.add_argument(
        "--delta-tail",
        type=int,
        nargs="+",
        default=[],
        metavar="N",
        help="add a cell per N: only the state's last N tokens change per decision (3.4)",
    )


def main(args: argparse.Namespace) -> int:
    from .mlx_runner import MLXRunner

    runner = MLXRunner.load(args.model, int8=args.int8)
    env = environment(args.model)
    rows = run_bench(
        runner, args.lengths, reps=args.reps, warmup=args.warmup, delta_tails=args.delta_tail
    )
    md, js = write_results(env, rows, args.out)
    print(md.read_text())
    print(f"wrote {md} and {js}")
    return 0
