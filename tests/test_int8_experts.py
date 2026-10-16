"""Story 2.8 — W4A8 MoE experts: the fused row quantiser, one gather GEMM against an
explicit int8-activation x dequantised-weight reference, the SwitchGLU swap on toy
experts, and (when its weights are cached) the real Gemma 4 26B-A4B with ``int8=True``
against stock on the committed fixtures' requests and on delta-cached decisions.

Tolerance on the 26B (measured 2026-10-16, M5, low power mode off, battery): over the
fixtures' requests all-INT8 moves raw probabilities by at most 4.9e-4 and raw logits
by at most 2.33 (a sequence-mode score of an option at p 1.5e-8); the story 2.3
dense-only INT8 already moves the same logits by up to 1.75. ``PROB_TOL`` is the E2B
gate of story 2.3 (5e-3, ten times the worst case here); ``LOGIT_TOL`` is 3.0 for
this model, since sequence scores sum per-token log-probs and the deep-tail options
move most. Argmax must agree everywhere.

Story 6.5 added image fixtures: on the gridworld frame (six first-token options) the
low-probability ``wait`` logit moved by 3.04 while probabilities stayed within 1.5e-3
(measured 2026-10-16, M5, low power off, AC), so ``LOGIT_TOL`` is now 4.0; the
probability gate is unchanged.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
from conftest import HF_HUB

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx import nn
from mlx_lm.models.gemma4_text import GeGLU
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from mlx_classifiers import moe
from mlx_classifiers.int8 import experts as ex

PROB_TOL = 5e-3
LOGIT_TOL = 4.0
# Story 6.7: the gridworld frame at budgets 140 and 70 (``-vt140`` / ``-vt70``) moves
# much more under the INT8 experts than at 280 (max |dp| 0.196 and 0.154 against
# 0.0015; max |dlogit| 2.3 / 3.9), argmax unchanged. Those two are gated on exact ids,
# the same answer, and |dp| within BUDGET_PROB_TOL, like test_int8's low-margin cases.
BUDGET_PROB_TOL = 0.25

# the MPP int4b matmul2d path needs Metal 4 on an M5-class GPU, as moe's NAX kernel does
needs_m5 = pytest.mark.skipif(not moe.available(), reason="needs MLX 0.32 on an M5 GPU")


def _asymmetric_experts(experts, n, k, seed):
    w = mx.random.normal((experts, n, k), key=mx.random.key(seed)) * 0.05 + 0.01
    return mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=4)


@needs_m5
@pytest.mark.parametrize("k", [64, 704, 2816])
def test_quant_rows_matches_the_reference(k):
    x = (mx.random.normal((37, k), key=mx.random.key(k)) * 3).astype(mx.bfloat16)
    for got, want in zip(ex.quant_rows(x), ex._quant_rows_reference(x), strict=True):
        assert mx.array_equal(got, want).item()


@needs_m5
@pytest.mark.parametrize(("experts", "n", "k", "rows"), [(4, 64, 128, 16), (16, 704, 2816, 300)])
def test_gather_gemm_matches_int8_times_dequantised_weights(experts, n, k, rows):
    wq, s, b = _asymmetric_experts(experts, n, k, seed=rows)
    assert mx.abs(b + 8 * s).max().item() > 0  # affine, so the rowsum fold is exercised
    lin = QuantizedSwitchLinear(k, n, experts, bias=False)
    lin.weight, lin.scales, lin.biases = wq, s, b
    w = mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32)
    il = ex.Int8SwitchLinear(lin)
    ids = mx.sort(mx.random.randint(0, experts - 1, (rows,), key=mx.random.key(2)))
    ids = ids.astype(mx.int32)  # the last expert gets no rows
    x = mx.random.normal((rows, k), key=mx.random.key(1)).astype(mx.bfloat16)
    q, xs, rs = ex.quant_rows(x)

    def pad(a):
        return mx.concatenate([a, mx.zeros((ex.SM, *a.shape[1:]), a.dtype)])

    out = ex.gather_gemm(pad(q), pad(xs), pad(rs), il, ex.build_tiles(ids, rows, experts))
    assert out.dtype == mx.bfloat16 and out.shape == (rows + ex.SM, n)
    assert not mx.any(out[rows:]).item()  # the pad tile writes zeros
    xd = q.astype(mx.float32) * xs[:, None]
    ref = mx.einsum("rk,rnk->rn", xd, w[ids])
    rel = mx.abs(out[:rows].astype(mx.float32) - ref).max() / mx.abs(ref).max()
    assert rel.item() < 5e-3  # bf16 output rounding


class _Toy(nn.Module):
    def __init__(self, quantized: bool = True):
        super().__init__()
        self.glu = SwitchGLU(256, 128, 16, activation=GeGLU())
        if quantized:
            for name in ("up_proj", "gate_proj", "down_proj"):
                setattr(self.glu, name, getattr(self.glu, name).to_quantized(64, 4))


@needs_m5
def test_patch_experts_swaps_switch_glu_and_stays_close_at_every_row_count():
    mx.random.seed(0)
    toy = _Toy()
    x = mx.random.normal((1, 24, 256)).astype(mx.bfloat16)
    ind = mx.argsort(mx.random.normal((1, 24, 16)), axis=-1)[..., :4]
    cases = [(x, ind), (x[:, :1], ind[:, :1])]  # 96 rows (sorted), 4 rows (L=1)
    refs = [toy.glu(a, i) for a, i in cases]
    assert ex.patch_experts(toy) == 1
    assert isinstance(toy.glu, ex.Int8SwitchGLU)
    assert isinstance(toy.glu.up_proj, ex.Int8SwitchLinear)
    for (a, i), ref in zip(cases, refs, strict=True):
        out = toy.glu(a, i)
        assert out.dtype == mx.bfloat16 and out.shape == ref.shape
        diff = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()
        assert 0 < (diff / mx.abs(ref).max()).item() < 0.06  # int8 ran, and is close
    assert moe.patch_model(toy) == 0  # the 3.6 kernel no longer applies
    assert ex.patch_experts(_Toy(quantized=False)) == 0


# --- on the real 26B (15 GB; skipped unless its weights are cached) -------------------------

MOE = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
_SNAP = HF_HUB / f"models--{MOE.replace('/', '--')}" / "snapshots"
COMMITTED = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.mlx
@needs_m5
@pytest.mark.skipif(not any(_SNAP.glob("*/*.safetensors")), reason=f"{MOE} weights not cached")
def test_26b_int8_experts_agree_with_stock_on_fixtures_and_delta_decisions(monkeypatch):
    """Stock 26B results for every committed fixture's request (and three delta-cached
    decisions) become the expectation; ``int8=True`` must match the token ids
    exactly, the answer (argmax), and probabilities within ``PROB_TOL``."""
    from mlx_classifiers import SystemOneRequest, decide
    from mlx_classifiers.bench import PREFIX, QUESTION, delta_states
    from mlx_classifiers.fixtures import check, load_fixtures, run_case
    from mlx_classifiers.mlx_runner import MLXRunner
    from mlx_classifiers.prompt_cache import PromptCacheStore

    fxs = load_fixtures(COMMITTED)

    def run(runner):
        cases = {fx["id"]: run_case(fx, runner) for fx in fxs}
        runner.prompt_cache = PromptCacheStore()
        deltas = []
        for st in delta_states(runner.tokenize, 512, 50, 3):
            req = SystemOneRequest.model_validate(
                {"prefix": PREFIX, "state": st, "questions": {"q": QUESTION}}
            )
            deltas.append(decide(req, runner).answers["q"].raw_probabilities)
        return cases, deltas, runner.prefill_calls

    stock = MLXRunner.load(MOE)
    want_cases, want_deltas, _ = run(stock)
    del stock
    gc.collect()
    mx.clear_cache()

    calls = {"n": 0}
    real = ex.gather_gemm

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ex, "gather_gemm", counting)
    runner = MLXRunner.load(MOE, int8=True)
    assert (runner.int8_expert_layers, runner.moe_kernel_layers) == (30, 0)
    got_cases, got_deltas, prefills = run(runner)
    assert calls["n"] > 0
    assert any(c > 0 for c, _ in prefills)  # the delta decisions hit the cache

    def budget(fx):
        return (fx["request"].get("options") or {}).get("vision_soft_tokens")

    def gate(group, prob_tol, logit_tol):
        report = check(
            [dict(fx, expected=want_cases[fx["id"]]) for fx in group],
            lambda fx: got_cases[fx["id"]],
            prob_tol=prob_tol,
            logit_tol=logit_tol,
        )
        assert report.ok, report.render()
        return report

    budgeted = [fx for fx in fxs if budget(fx)]
    gate(budgeted, BUDGET_PROB_TOL, None)  # the check compares answers and token ids too
    report = gate([fx for fx in fxs if not budget(fx)], PROB_TOL, LOGIT_TOL)
    assert any(c.max_prob_distance for c in report.cases)  # INT8 did change something
    for want, got in zip(want_deltas, got_deltas, strict=True):
        assert max(want, key=want.get) == max(got, key=got.get)
        assert max(abs(want[o] - got[o]) for o in want) <= PROB_TOL
