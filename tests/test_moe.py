"""Story 3.6 — the per-expert MoE gather-QMM: dispatch window, bit-identity against
``mx.gather_qmm`` on synthetic experts, the SwitchGLU swap, and (when its weights
are cached) identical delta-cached decisions on the real Gemma 4 26B-A4B."""

from __future__ import annotations

import pytest
from conftest import HF_HUB

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx import nn
from mlx_lm.models.switch_layers import SwitchGLU

from mlx_classifiers import moe

needs_nax = pytest.mark.skipif(not moe.available(), reason="needs MLX 0.32 on a NAX GPU (M5)")


def test_use_segmented_window():
    assert not moe.use_segmented(511, 128)  # MLX's per-row path: different rounding
    assert moe.use_segmented(512, 128) and moe.use_segmented(776, 128)  # ~97-token delta
    assert moe.use_segmented(2048, 128)
    assert not moe.use_segmented(2049, 128)  # long chunks: stock 64-row tiles win
    assert not moe.use_segmented(8, 128)  # the L=1 readout pass


def _quantized_experts(experts, n, k, seed):
    w = (mx.random.normal((experts, n, k), key=mx.random.key(seed)) * 0.05).astype(mx.bfloat16)
    return mx.quantize(w, group_size=64, bits=4)


@needs_nax
@pytest.mark.parametrize(("experts", "n", "k", "rows"), [(16, 128, 256, 64), (16, 256, 128, 250)])
def test_segmented_gather_qmm_is_bit_identical_to_gather_qmm(experts, n, k, rows):
    wq, s, b = _quantized_experts(experts, n, k, seed=rows)
    x = mx.random.normal((rows, k), key=mx.random.key(1)).astype(mx.bfloat16)
    ids = mx.sort(mx.random.randint(0, experts - 2, (rows,), key=mx.random.key(2)))
    ids = ids.astype(mx.uint32)  # the last two experts get no rows
    ref = mx.gather_qmm(
        x[:, None], wq, s, b, rhs_indices=ids, transpose=True, group_size=64, bits=4,
        sorted_indices=True,
    )[:, 0]  # fmt: skip
    got = moe.segmented_gather_qmm(x, wq, s, b, moe.expert_offsets(ids, experts))
    assert mx.array_equal(ref, got).item()


class _Toy(nn.Module):
    def __init__(self, quantized: bool):
        super().__init__()
        self.glu = SwitchGLU(256, 128, 16)
        if quantized:
            for name in ("up_proj", "gate_proj", "down_proj"):
                setattr(self.glu, name, getattr(self.glu, name).to_quantized(64, 4))


@needs_nax
def test_patch_model_swaps_quantized_switch_glu_and_keeps_outputs():
    mx.random.seed(0)
    toy = _Toy(quantized=True)
    x = mx.random.normal((1, 24, 256)).astype(mx.bfloat16)
    inside = mx.random.randint(0, 16, (1, 24, 4))  # 96 rows over 16 experts: in the window
    outside = inside[:, :2]  # 8 rows: stock path
    ref_in, ref_out = toy.glu(x, inside), toy.glu(x[:, :2], outside)
    assert moe.patch_model(toy) == 1
    assert isinstance(toy.glu, moe.SegmentedSwitchGLU)
    assert mx.array_equal(toy.glu(x, inside), ref_in).item()
    assert mx.array_equal(toy.glu(x[:, :2], outside), ref_out).item()
    assert moe.patch_model(_Toy(quantized=False)) == 0  # unquantized experts: left alone


# --- on the real 26B (15 GB; skipped unless its weights are cached) -------------------------

MOE = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
_SNAP = HF_HUB / f"models--{MOE.replace('/', '--')}" / "snapshots"


@pytest.mark.mlx
@needs_nax
@pytest.mark.skipif(not any(_SNAP.glob("*/*.safetensors")), reason=f"{MOE} weights not cached")
def test_26b_delta_decisions_are_identical_with_the_kernel_on_and_off():
    from mlx_classifiers import SystemOneRequest, decide
    from mlx_classifiers.bench import PREFIX, QUESTION, delta_states
    from mlx_classifiers.mlx_runner import MLXRunner
    from mlx_classifiers.prompt_cache import PromptCacheStore

    runner = MLXRunner.load(MOE)
    assert runner.moe_kernel_layers == 30
    glus = [m for _, m in runner.model.named_modules() if isinstance(m, moe.SegmentedSwitchGLU)]
    states = delta_states(runner.tokenize, 512, 50, 3)

    def run():
        runner.prompt_cache = PromptCacheStore()
        out = []
        for st in states:  # first decision fills the store, the next two are delta hits
            req = SystemOneRequest.model_validate(
                {"prefix": PREFIX, "state": st, "questions": {"q": QUESTION}}
            )
            out.append(decide(req, runner).answers["q"].raw_probabilities)
        return out

    on = run()
    # the delta hits prefilled ~100 tokens (~800 expert rows): inside the kernel's window
    assert any(c > 0 and 64 <= n <= 256 for c, n in runner.prefill_calls)
    for m in glus:
        m.__class__ = SwitchGLU
    off = run()
    assert on == off
