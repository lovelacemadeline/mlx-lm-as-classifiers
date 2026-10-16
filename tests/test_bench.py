"""Story 2.4 — benchmark harness, exercised on the FakeRunner (no weights)."""

from __future__ import annotations

import json

from mlx_classifiers.bench import (
    environment,
    filler_state,
    render_markdown,
    run_bench,
    write_results,
)
from mlx_classifiers.runner import FakeRunner


def test_filler_state_hits_the_token_budget():
    runner = FakeRunner()
    for n in (16, 64, 200):
        text, count = filler_state(runner.tokenize, n)
        assert count == len(runner.tokenize(text)) and n - 3 <= count <= n


def test_run_bench_sweeps_lengths_and_cache(tmp_path):
    runner = FakeRunner()
    rows = run_bench(runner, lengths=[16, 48], reps=3, warmup=1)
    assert [(r.state_tokens <= 48, r.prefix_cached) for r in rows] == [
        (True, False),
        (True, True),
        (True, False),
        (True, True),
    ]
    uncached, cached = rows[2], rows[3]
    assert cached.prefilled_tokens < uncached.prefilled_tokens == uncached.prompt_tokens
    for r in rows:
        assert r.p95_ms >= r.p50_ms > 0
        assert r.decisions_per_s > 0 and r.prefill_tok_per_s > 0

    env = environment("fake")
    assert {"model", "git_rev", "lowpowermode", "power_source", "machine"} <= set(env)
    md = render_markdown(env, rows)
    assert "| state tok |" in md and "decisions/s" in md and "power:" in md
    md_path, js_path = write_results(env, rows, tmp_path)
    data = json.loads(js_path.read_text())
    assert data["env"]["model"] == "fake" and len(data["rows"]) == 4
    assert md_path.read_text() == md
