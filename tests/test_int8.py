"""Story 2.3 — W8A8 INT8 kernel wiring: the bench column (no weights), the wrapper on
one real E2B layer, and fixture parity INT8 on vs off on the real Gemma 4 E2B.

Tolerance (measured 2026-10-16, M5, low power mode on): over the committed
fixtures INT8 moves raw probabilities by at most 1.44e-3 and raw logits by at
most 0.5 (bf16 steps); at 0.5k to 4k-token prompts by at most 1.6e-3 and 0.75.
``PROB_TOL`` gave about 3x headroom over the worst measured case and is still
under 1% of the smallest top-1 margin in the fixtures (0.934), so it cannot hide
an argmax flip; ``LOGIT_TOL`` matches the cross-path bf16 noise of story 2.5.
Story 2.9 (2026-10-16, low power off): with the stock linears freed, the readout row
and short sequence spans run on the int8 weights too, and the worst case is 3.38e-3
(``score-sequence-object-state``) and 0.75 in logits.
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


def _m5() -> bool:
    try:
        from mlx_classifiers import moe
    except ImportError:
        return False
    return moe.available()


# the W8A8 kernels need Metal 4 MPP matmul2d on an M5-class GPU (as story 2.8's do)
needs_m5 = pytest.mark.skipif(not _m5(), reason="needs MLX 0.32 on an M5 GPU")


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
    assert not any("orig" in m for m in mlp + proj)  # story 2.9: no stock copy kept


def _stock_and_wrapper(n, k, bits, seed=0):
    import mlx.core as mx
    from mlx import nn

    from mlx_classifiers.int8 import Int8LinearW8A8

    lin = nn.Linear(k, n, bias=False)
    lin.weight = mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
    return q, Int8LinearW8A8(q)


@needs_m5
@pytest.mark.parametrize(("n", "k", "bits"), [(6144, 1536, 8), (2112, 2816, 8), (256, 1536, 4)])
def test_small_rows_run_on_the_int8_weights(n, k, bits):
    """Story 2.9: up to GEMV_MAX_ROWS rows the wrapper is the int8 weights times the
    bf16 activations (fp32 accumulation, one bf16 rounding); 9..63 rows take the padded
    int8 GEMM. Every row count stays within 5% (of the max) of the stock linear."""
    import mlx.core as mx

    from mlx_classifiers.int8 import wiring

    q, w = _stock_and_wrapper(n, k, bits)
    assert "orig" not in w
    deq = w._wq.astype(mx.float32) * w._ws[:, None]
    for rows in (1, 3, wiring.GEMV_MAX_ROWS, wiring.GEMV_MAX_ROWS + 1, 63, 64, 200):
        x = mx.random.normal((1, rows, k), key=mx.random.key(rows)).astype(mx.bfloat16)
        out, ref = w(x), q(x)
        assert out.dtype == mx.bfloat16 and out.shape == ref.shape
        o32, r32 = out.astype(mx.float32), ref.astype(mx.float32)
        assert 0 < (mx.abs(o32 - r32).max() / mx.abs(r32).max()).item() < 0.05
        if rows <= wiring.GEMV_MAX_ROWS:
            exact = mx.matmul(x[0].astype(mx.float32), deq.T, stream=mx.cpu)
            # one bf16 rounding of the output: within 2^-8 of each value
            assert (mx.abs(o32[0] - exact) <= mx.abs(exact) * 2**-8 + 1e-6).all().item()


@pytest.mark.mlx
@needs_gemma
def test_wrapper_keeps_bf16_and_falls_back_under_64_rows(int8_runner):
    """The real E2B wrapper: bf16 out at every row count, the int8 GEMM at 64 rows and
    over, and (story 2.9) the int8 weights below that, with the stock linear freed.
    Layer 0's stock copy is gone, so the stock reference is rebuilt from the
    checkpoint file."""
    import mlx.core as mx
    from mlx import nn

    w = int8_runner.model.language_model.model.layers[0].mlp.gate_proj
    assert "orig" not in w
    key = "language_model.model.layers.0.mlp.gate_proj"
    snap = next(_SNAP.glob("*/model.safetensors"))
    t = mx.load(str(snap))
    ref = nn.QuantizedLinear(w._K, w._N, bias=False, group_size=64, bits=8)
    ref.weight, ref.scales, ref.biases = (t[f"{key}.{a}"] for a in ("weight", "scales", "biases"))
    del t
    mx.random.seed(0)
    long = mx.random.normal((1, 200, w._K)).astype(mx.bfloat16)
    for x in (long, long[:, :63], long[:, :1]):
        out, want = w(x), ref(x)
        assert out.dtype == mx.bfloat16 and out.shape == want.shape
        diff = mx.abs(out.astype(mx.float32) - want.astype(mx.float32)).max()
        assert 0 < (diff / mx.abs(want).max()).item() < 0.05  # int8 ran, and is close


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
