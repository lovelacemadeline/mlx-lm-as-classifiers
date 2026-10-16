# Vendored for mlx-lm-as-classifiers story 2.8 on 2026-10-16 from two sources:
#   (1) /Users/maddie/Documents/dev/2026/aug/ling3-mlx/moe_gather_int8.py
#       git 2b97d6f7df728dd441bef2f2805f564511baf801 (2026-08-11),
#       sha256 dd80057c470ea271a3951874ac55eb2aa4b11550709a090ca5172268f4bae742
#       -- the kernel skeleton: int8 x int4b_format matmul2d reading the
#       checkpoint's (E, N, K) nibbles in place (transpose_right, XOR 0x8 repack),
#       16-row M tiles, device-side tile table with a zero pad tile, bf16 output.
#   (2) /Users/maddie/Documents/dev/2026/july/int8-mlx/agent_e5_affine_gather.py
#       git f924f0fb82bdcef719b8dcd2732ddb5f0aeb1e53 (2026-07-31),
#       sha256 893a4b000bfb16b788ef49c24001267520e4580e2c40c33cce0ac1231f5259b3
#       -- the affine group-64 epilogue (int8-mlx story 5.3/5.17): out = xs *
#       sum_g [ s*int(xq . (u-8)) + (b + 8s) * rowsum_g(xq) ], exact against
#       mx.dequantize, so the expert weights are used bit-exact.
# (1) assumes a symmetric group-32 checkpoint; the Gemma 4 26B-A4B qat-4bit
# experts are affine group 64 (34% of groups have b != -8s), hence (2)'s
# rowsum fold, applied once per two 32-wide K steps. The 5.21 fix (bf16 out,
# never fp16 into a bf16 stream) is kept. Needs Apple M5 / Metal 4 (MPP
# matmul2d cooperative tensors with int4b_format).
"""W4A8 expert GEMM for 4-bit affine MoE experts (Gemma 4 26B-A4B), story 2.8.

:func:`patch_experts` swaps every eligible ``SwitchGLU`` (4-bit affine, group 64,
no bias, 64-aligned dims) for :class:`Int8SwitchGLU`, whose three projections are
:class:`Int8SwitchLinear`: the checkpoint's nibbles XOR'd to signed ``int4b``
(same bytes), scales and biases transposed to ``(E, K/64, N)`` (``b + 8s`` is formed
in the kernel, story 2.9). The
stock weights are dropped, so this **replaces** the expert path at every row
count, including the L=1 readout, and :mod:`mlx_classifiers.moe`'s bit-identical
kernel (which needs the stock encoding) no longer applies to that model.

Per call: activations are quantised per token to symmetric int8 (with per-group
row sums) before the gather, the gathered rows are sorted by expert, and one
threadgroup runs one 16-row tile of one expert against one 64-column block.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

__all__ = [
    "Int8SwitchGLU",
    "Int8SwitchLinear",
    "build_tiles",
    "gather_gemm",
    "patch_experts",
    "quant_rows",
]

_HEADER = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

SK = 32  # K step; two per affine group
SN = 64  # column block per threadgroup
SM = 16  # rows per tile
GROUP = 64
THREADS = 32  # one simdgroup per threadgroup
_XOR = 0x88888888  # unsigned offset-8 nibble u -> two's-complement (u - 8)


@cache
def _kernel(sm: int = SM) -> Any:
    cap = sm * SN // THREADS
    src = f"""
    constexpr int SM = {sm}, SN = {SN}, SK = {SK}, GROUP = {GROUP};
    const int N  = int(dims[0]);
    const int K  = int(dims[1]);
    const int NG = K / GROUP;

    const int t  = int(threadgroup_position_in_grid.x);
    const int nr = tile_nrows[t];
    if (nr <= 0) return;                 // dead tile from the static grid
    const int e    = tile_expert[t];
    const int row0 = tile_row0[t];
    const int col  = int(threadgroup_position_in_grid.y) * SN;

    const device uchar* Bb = (const device uchar*)B;
    const size_t Bbase = (size_t)e * N * K;     // in nibbles
    const size_t Sbase = (size_t)e * NG * N;    // (E, NG, N)

    constexpr auto desc = matmul2d_descriptor(
        SM, SN, SK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    using tA_t = tensor<device int8_t,       extents<int32_t, SK, SM>, tensor_inline>;
    using tB_t = tensor<device int4b_format, extents<int32_t, SK, SN>, tensor_inline>;

    auto ct_a = op.get_left_input_cooperative_tensor<int8_t, int4b_format, int>();
    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), tB_t, int>();
    float acc[{cap}];
    for (int i = 0; i < {cap}; ++i) acc[i] = 0.0f;
    for (ushort i = 0; i < ct_c.get_capacity(); ++i) ct_c.set(i, 0);

    for (int g = 0; g < NG; ++g) {{
      for (int kk = 0; kk < GROUP; kk += SK) {{
        const int k = g * GROUP + kk;
        tA_t sA((device int8_t*)(A + (size_t)row0 * K + k),
                extents<int32_t, SK, SM>(), array<int32_t, 2>{{1, K}});
        ct_a.load(sA);
        tB_t sB((device uchar*)(Bb + (Bbase + (size_t)col * K + k) / 2),
                extents<int32_t, SK, SN>(), array<int32_t, 2>{{1, K}});
        op.run(ct_a, sB, ct_c);
      }}
      // affine fold for group g: s * int(xq . (u - 8)) + (b + 8s) * rowsum_g(xq)
      for (ushort i = 0; i < ct_c.get_capacity(); ++i) {{
        auto mi = ct_c.get_multidimensional_index(i);
        const int c = col + mi[0];
        const size_t si = Sbase + (size_t)g * N + c;
        const float rs = (float)RS[(size_t)(row0 + mi[1]) * NG + g];
        const float sc = (float)S[si];
        acc[i] += (float)ct_c[i] * sc + ((float)BIAS[si] + 8.0f * sc) * rs;
        ct_c.set(i, 0);
      }}
    }}
    for (ushort i = 0; i < ct_c.get_capacity(); ++i) {{
      auto mi = ct_c.get_multidimensional_index(i);
      const int rl = mi[1];
      if (rl < nr) {{
        const int r = row0 + rl;
        D[(size_t)r * N + col + mi[0]] = (bfloat)(acc[i] * a_scale[r]);
      }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"mlx_classifiers_w4a8_affine_gather_sm{sm}",
        input_names=["A", "B", "S", "BIAS", "a_scale", "RS", "tile_expert", "tile_row0"]
        + ["tile_nrows", "dims"],
        output_names=["D"],
        header=_HEADER,
        source=src,
    )


def build_tiles(idx_sorted: Any, rows: int, experts: int, sm: int = SM) -> tuple[Any, ...]:
    """Device-side tile table ``(expert, row0, nrows, n_tiles)`` for ``rows`` sorted ids.

    The grid size depends only on ``rows`` (no readback); surplus tiles have
    ``nrows == 0`` and exit at once. One extra tile covers the ``sm`` zero pad rows
    after ``rows`` so the output's pad rows are written (zeros), which lets the
    down projection read the up/gate output directly. From source (1); the per-expert
    counts are a scatter-add instead of a (rows, experts) comparison."""
    maxt = (rows + sm - 1) // sm + experts
    counts = mx.zeros((experts,), mx.int32).at[idx_sorted].add(1)
    starts = mx.cumsum(counts) - counts
    ntiles_e = (counts + sm - 1) // sm
    tile_end = mx.cumsum(ntiles_e)
    tstart_e = tile_end - ntiles_e
    t = mx.arange(maxt)
    e = mx.minimum(mx.sum((t[:, None] >= tile_end[None, :]).astype(mx.int32), 1), experts - 1)
    local = t - tstart_e[e]
    return (
        mx.concatenate([e.astype(mx.int32), mx.zeros((1,), mx.int32)]),
        mx.concatenate([(starts[e] + local * sm).astype(mx.int32), mx.array([rows], mx.int32)]),
        mx.concatenate(
            [mx.clip(counts[e] - local * sm, 0, sm).astype(mx.int32), mx.array([sm], mx.int32)]
        ),
        maxt + 1,
    )


_QUANT_SRC = """
    constexpr int NT = 256;
    const int K = int(dims[0]);
    const int NG = K / 64;
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32u;
    const uint sg = tid / 32u;
    auto xr = x + (size_t)row * K;
    device int8_t* cr = codes + (size_t)row * K;
    threadgroup float smax[NT / 32];
    threadgroup int half_sum[NT / 32];

    float m = 1e-8f;
    for (int i = int(tid); i < K; i += NT) m = max(m, fabs((float)xr[i]));
    m = simd_max(m);
    if (lane == 0) smax[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < NT / 32) ? smax[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) smax[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float amax = smax[0];
    const float inv = 127.0f / amax;
    if (tid == 0) scales[row] = amax / 127.0f;

    // a 256-wide stripe is four 64-wide groups; simdgroups 2j and 2j+1 cover group j
    for (int base = 0; base < K; base += NT) {
        const int i = base + int(tid);
        int c = 0;
        if (i < K) {
            c = int(clamp(rint((float)xr[i] * inv), -127.0f, 127.0f));
            cr[i] = (int8_t)c;
        }
        const int hs = simd_sum(c);
        if (lane == 0) half_sum[sg] = hs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) {
            const int g = base / 64 + int(tid);
            if (g < NG) rowsum[(size_t)row * NG + g] = half_sum[2 * tid] + half_sum[2 * tid + 1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


@cache
def _quant_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="mlx_classifiers_w4a8_rowquant_rowsum",
        input_names=["x", "dims"],
        output_names=["codes", "scales", "rowsum"],
        source=_QUANT_SRC,
    )


def quant_rows(x: Any) -> tuple[Any, Any, Any]:
    """``x`` (rows, K) -> symmetric per-row int8 codes, fp32 scales (rows,), and the
    int32 code sums of each 64-wide group (rows, K/64) for the affine fold. One
    fused pass (the per-row quantiser of :mod:`.wiring` plus the group sums)."""
    rows, k = x.shape
    return _quant_kernel()(
        inputs=[x, mx.array([k], mx.uint32)],
        output_shapes=[(rows, k), (rows,), (rows, k // GROUP)],
        output_dtypes=[mx.int8, mx.float32, mx.int32],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )


def _quant_rows_reference(x: Any) -> tuple[Any, Any, Any]:
    """Plain-mx equivalent of :func:`quant_rows` (tests)."""
    xf = x.astype(mx.float32)
    amax = mx.maximum(mx.max(mx.abs(xf), axis=-1), 1e-8)
    q = mx.clip(mx.round(xf * (127.0 / amax)[:, None]), -127, 127)
    rs = q.reshape(q.shape[0], -1, GROUP).sum(-1).astype(mx.int32)
    return q.astype(mx.int8), amax / 127.0, rs


def gather_gemm(aq: Any, a_s: Any, a_rs: Any, lin: Int8SwitchLinear, tiles: tuple) -> Any:
    """Padded sorted rows ``aq`` (R + SM, K) through ``lin`` -> (R + SM, N) bf16."""
    te, tr, tn, maxt = tiles
    n, k = lin.output_dims, lin.input_dims
    (d,) = _kernel(SM)(
        inputs=[aq, lin.w4, lin.s, lin.b, a_s, a_rs, te, tr, tn, mx.array([n, k], mx.uint32)],
        output_shapes=[(aq.shape[0], n)],
        output_dtypes=[mx.bfloat16],
        grid=(maxt * THREADS, n // SN, 1),
        threadgroup=(THREADS, 1, 1),
    )
    return d


class Int8SwitchLinear(nn.Module):
    """One expert projection in kernel form, built from a ``QuantizedSwitchLinear``
    (4-bit affine, group 64): signed-nibble weights (same bytes), the checkpoint's
    scales and biases in their own dtype, both transposed to (E, K/64, N). The kernel
    forms ``b + 8s`` in fp32 per group (exact: ``8s`` is a power-of-two scaling, so
    the sum rounds once, as the fp32 precompute of story 2.8 did; story 2.9 dropped
    that precompute, which doubled the fold operands to fp32). The stock arrays are
    not kept."""

    def __init__(self, orig: QuantizedSwitchLinear):
        super().__init__()
        self.num_experts = orig.num_experts
        self.input_dims, self.output_dims = orig.input_dims, orig.output_dims
        self.w4 = orig["weight"] ^ mx.array(_XOR, mx.uint32)
        self.s = mx.contiguous(mx.swapaxes(orig["scales"], 1, 2))
        self.b = mx.contiguous(mx.swapaxes(orig["biases"], 1, 2))
        mx.eval(self.w4, self.s, self.b)


class Int8SwitchGLU(SwitchGLU):
    """``SwitchGLU`` over :class:`Int8SwitchLinear` projections (all row counts)."""

    def __call__(self, x: Any, indices: Any) -> Any:
        shape, topk = indices.shape, indices.shape[-1]
        h = x.shape[-1]
        flat = x.reshape(-1, h)
        t = flat.shape[0]
        idx = indices.flatten()
        order = mx.argsort(idx)
        inv_order = mx.argsort(order)
        rows = idx.size
        # quantise once per token, then gather codes; row t is the zero pad row
        xq, xs, xr = quant_rows(flat)
        xq = mx.concatenate([xq, mx.zeros((1, h), mx.int8)])
        xs = mx.concatenate([xs, mx.zeros((1,), mx.float32)])
        xr = mx.concatenate([xr, mx.zeros((1, xr.shape[1]), mx.int32)])
        src = mx.concatenate([order // topk, mx.full((SM,), t, order.dtype)])
        aq, a_s, a_rs = xq[src], xs[src], xr[src]
        tiles = build_tiles(idx[order], rows, self.up_proj.num_experts)
        up = gather_gemm(aq, a_s, a_rs, self.up_proj, tiles)
        gate = gather_gemm(aq, a_s, a_rs, self.gate_proj, tiles)
        hq, hs, hr = quant_rows(self.activation(up, gate))  # pad rows stay zero
        out = gather_gemm(hq, hs, hr, self.down_proj, tiles)
        return mx.unflatten(out[inv_order], 0, tuple(shape)).astype(x.dtype)


def _eligible(lin: Any) -> bool:
    return (
        isinstance(lin, QuantizedSwitchLinear)
        and getattr(lin, "mode", "affine") == "affine"
        and lin.bits == 4
        and lin.group_size == GROUP
        and lin.get("biases") is not None
        and "bias" not in lin
        and lin.output_dims % SN == 0
        and lin.input_dims % GROUP == 0
    )


def patch_experts(model: Any) -> int:
    """Swap every eligible ``SwitchGLU`` in ``model`` for :class:`Int8SwitchGLU`,
    converting its projections in place (stock expert weights released, and their
    buffers returned by ``mx.clear_cache``, story 2.9). Returns the number swapped;
    0 on a dense model."""
    targets = [
        m
        for _, m in model.named_modules()
        if isinstance(m, SwitchGLU)
        and not isinstance(m, Int8SwitchGLU)
        and all(_eligible(getattr(m, p)) for p in ("up_proj", "gate_proj", "down_proj"))
    ]
    for m in targets:
        for name in ("up_proj", "gate_proj", "down_proj"):
            setattr(m, name, Int8SwitchLinear(getattr(m, name)))
        m.__class__ = Int8SwitchGLU
    mx.clear_cache()
    return len(targets)
