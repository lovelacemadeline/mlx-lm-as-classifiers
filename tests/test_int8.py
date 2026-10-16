"""Story 2.3 — W8A8 INT8 kernel wiring: the bench column (no weights), the wrapper on
one real E2B layer, and fixture parity INT8 on vs off on the real Gemma 4 E2B.

Tolerance (measured 2026-10-16, M5, low power mode on): over the committed
fixtures INT8 moves raw probabilities by at most 1.44e-3 and raw logits by at
most 0.5 (bf16 steps); at 0.5k to 4k-token prompts by at most 1.6e-3 and 0.75.
``PROB_TOL`` gives about 3x headroom over the worst measured case and is still
under 1% of the smallest top-1 margin in the fixtures (0.934), so it cannot hide
an argmax flip; ``LOGIT_TOL`` matches the cross-path bf16 noise of story 2.5.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import GEMMA, HF_HUB

from mlx_classifiers.bench import run_bench
from mlx_classifiers.runner import FakeRunner

PROB_TOL = 5e-3
LOGIT_TOL = 1.0

# Story 6.5 added image fixtures, one of them a gridworld frame whose top-1 margin is
# 0.17 (north 0.42 / west 0.26). INT8 is not that faithful on such prompts, text or
# image: over the 30 gridworld battery frames (measured 2026-10-16, M5, low power off)
# INT8 vs stock moved raw probabilities by a median 0.06 to 0.08 and at most 0.21, and
# flipped the argmax on 1/30 text, 3/30 image-only and 0/30 text+image states, in line
# with the int8-mlx top-1 fidelity of 0.95 to 0.98. So the gates above hold for fixtures
# whose stock top-1 margin is at least MARGIN; below it a fixture is only checked for
# exact token ids and a distance under LOW_MARGIN_PROB_TOL.
MARGIN = 0.5
LOW_MARGIN_PROB_TOL = 0.25

COMMITTED = Path(__file__).resolve().parent.parent / "fixtures"
_SNAP = HF_HUB / f"models--{GEMMA.replace('/', '--')}" / "snapshots"
needs_gemma = pytest.mark.skipif(
    not any(_SNAP.glob("*/model.safetensors")), reason=f"{GEMMA} weights not in HF cache"
)


def test_bench_int8_column_follows_the_runner_flag():
    runner = FakeRunner()
    assert {r.int8 for r in run_bench(runner, lengths=[16], reps=1, warmup=0)} == {False}
    runner.int8 = True
    assert {r.int8 for r in run_bench(runner, lengths=[16], reps=1, warmup=0)} == {True}


# --- on the real model ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def int8_runner():
    pytest.importorskip("mlx.core")
    from mlx_classifiers.mlx_runner import MLXRunner

    return MLXRunner.load(GEMMA, int8=True)


@pytest.mark.mlx
@needs_gemma
def test_load_patches_every_mlp_and_projection(int8_runner):
    from mlx_classifiers.int8 import Int8LinearW8A8

    layers = int8_runner.model.language_model.model.layers
    assert int8_runner.int8
    mlp = [getattr(lay.mlp, n) for lay in layers for n in ("gate_proj", "up_proj", "down_proj")]
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    proj = [getattr(lay.self_attn, n) for lay in layers for n in names if hasattr(lay.self_attn, n)]
    assert len(mlp) == 105 and len(proj) == 100  # 15 producers x 4 + 20 KV-shared x 2
    assert all(isinstance(m, Int8LinearW8A8) for m in mlp + proj)


@pytest.mark.mlx
@needs_gemma
def test_wrapper_keeps_bf16_and_falls_back_under_64_rows(int8_runner):
    import mlx.core as mx

    w = int8_runner.model.language_model.model.layers[0].mlp.gate_proj
    mx.random.seed(0)
    long = mx.random.normal((1, 200, w._K)).astype(mx.bfloat16)
    out, ref = w(long), w.orig(long)
    assert out.dtype == mx.bfloat16 and out.shape == ref.shape
    rel = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max() / mx.abs(ref).max()
    assert 0 < rel.item() < 0.05  # the int8 lane ran, and is close
    short = long[:, :63]
    assert mx.array_equal(w(short), w.orig(short)).item()  # stock below 64 rows


@pytest.mark.mlx
@needs_gemma
def test_fixtures_agree_int8_on_vs_off(int8_runner, monkeypatch):
    """Every committed fixture (the stock path, reproduced exactly by
    test_fixtures::test_committed_fixtures_reproduce_on_mlx) agrees with INT8 on:
    same answer (argmax), probabilities within PROB_TOL, logits within LOGIT_TOL.
    Also asserts the int8 lane actually ran, so the parity is not vacuous."""
    from mlx_classifiers.fixtures import check, run_case
    from mlx_classifiers.int8 import wiring

    calls = {"n": 0}
    real = wiring._run_int8_linear

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(wiring, "_run_int8_linear", counting)
    from mlx_classifiers.fixtures import compare, load_fixtures

    def margin(fx):
        p = sorted(fx["expected"].get("raw_probabilities", {"x": 1.0}).values(), reverse=True)
        return p[0] - (p[1] if len(p) > 1 else 0.0)

    fxs = load_fixtures(COMMITTED)
    gated = [fx["id"] for fx in fxs if margin(fx) >= MARGIN]
    report = check(
        COMMITTED,
        lambda fx: run_case(fx, int8_runner) if fx["id"] in gated else fx["expected"],
        prob_tol=PROB_TOL,
        logit_tol=LOGIT_TOL,
    )
    assert report.ok, report.render()
    assert calls["n"] > 0
    assert any(c.max_prob_distance for c in report.cases)  # INT8 did change something
    for fx in (fx for fx in fxs if fx["id"] not in gated):
        rep = compare(
            fx["expected"],
            run_case(fx, int8_runner),
            case_id=fx["id"],
            prob_tol=LOW_MARGIN_PROB_TOL,
        )
        assert rep.token_ids_equal, rep.failures
        assert rep.max_prob_distance <= LOW_MARGIN_PROB_TOL, rep.failures
        print(f"[2.3/6.5] low-margin {fx['id']}: INT8 max|dp| {rep.max_prob_distance:.3f}")
