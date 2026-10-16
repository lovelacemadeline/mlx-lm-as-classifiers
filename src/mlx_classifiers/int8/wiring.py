# Vendored for mlx-lm-as-classifiers story 2.3 on 2026-10-16 from
#   /Users/maddie/Documents/dev/2026/aug/poorly-ported-gemma4-int8-optimisations/w8a8_wiring.py
#   (mtime 2026-08-09, sha256 7f7be1111c23a34704666935582b93e822f9dbef5fcd93e1bf746f06963159e2)
# Kernel sources and numerics are unchanged. Local changes, all in the patch
# entry points at the bottom: the layer walk goes through `_decoder_layers`
# (accepts the gemma4 multimodal wrapper or a bare text model), and
# `patch_model` applies both patches. Run on Gemma 4 E2B (35 layers; layers
# 15..34 are KV-shared and have no k_proj/v_proj, which `patch_projs` already
# skips via hasattr). Needs Apple M5 / Metal 4 (MPP matmul2d cooperative tensors).
# Story 2.9 (2026-10-16) changed the small-row path and the patch entry points: the
# wrapper no longer keeps the stock linear. Calls of <= GEMV_MAX_ROWS rows (the L=1
# readout) run `_gemv_int8`, a GEMV over the int8 weights dequantised on the fly
# (bf16 activations, fp32 accumulation); 9..63 rows are zero-padded to 64 and take
# the int8 GEMM. `patch_mlps`/`patch_projs` only wrap linears the kernels can serve
# (`_eligible`) and `patch_model` frees the dropped stock weights (`mx.clear_cache`).
# The int8 weight is stored (N, K) and the GEMM reads it with transpose_right
# (bit-equal output to the (K, N) original), and the requantisation runs in mx
# instead of numpy (bit-equal weights). The GEMM's numerics, the row quantiser and
# the 5.21 dtype fix are the source's; formatting is off to keep it diffable.
# ruff: noqa
# fmt: off
"""W8A8 dense-MLP + attention-projection wiring for Gemma4 MoE prefill.

Ported (logic-adapted, no cross-repo import) from the read-only reference
project at /Users/maddie/Documents/dev/2026/july/int8-mlx:
  - phase9_e2e_prefill.py  (Int8LinearW8A8, patch_mlps) -- W8A8 dense MLP
  - agent_91_wire.py       (patch_projs)                -- W8A8 attn q/k/v/o
  - phase5_int8linear.py   (metal_kernel matmul2d GEMM with scale epilogue)
  - fused_rowquant.py      (fused per-row dynamic int8 activation quant)

Scheme:
  - weights: dequantized from QAT-4bit checkpoint -> fp32 -> per-output-channel
    (per row of the (N, K) weight, i.e. per output feature) SYMMETRIC int8,
    pre-transposed to (K, N) so the GEMM kernel stays NN.
  - activations: dynamic per-token (per-row of the flattened (M, K) input)
    SYMMETRIC int8, computed in a single fused Metal kernel pass at call time.
  - GEMM: custom Metal kernel (MPP matmul2d cooperative-tensor path) with an
    epilogue that dequantizes int32 accumulator -> fp16 using the rank-1
    activation/weight scales.
  - Fallback: decode-shaped or misaligned calls (M < 64 rows, K not a
    multiple of 32, or N not a multiple of 64) fall back to the ORIGINAL
    (stock quantized) linear -- this is what keeps decode-time behavior/
    accuracy untouched; only prefill-shaped calls take the int8 fast lane.
    [mlx-classifiers story 2.9: no longer. Misaligned linears are not wrapped,
    and short calls run on the int8 weights; see the header above.]

Known gotcha (int8-mlx story 5.21, "the +1160ms dtype-promotion autopsy"):
  MLX promotes `bf16 (+|*) fp16 -> float32`. The GEMM kernel's epilogue
  hardcodes fp16 output; if that fp16 tensor is returned into a model whose
  residual stream is bf16 (or generally not fp16), every downstream op on
  that branch silently promotes to fp32 for the REST of the model (2x
  activation bytes + fp32 kernel variants) -- a large, easily-missed tax
  that has nothing to do with the int8 GEMM itself. The fix is one line:
  cast the wrapper's output back to the *input's* dtype at the call
  boundary. This is preserved below (`.astype(x.dtype)` in
  Int8LinearW8A8.__call__) exactly as story 5.21 fixed it -- do not drop it.

This module intentionally has NO import of anything under
/Users/maddie/Documents/dev/2026/july/int8-mlx -- it is a clean, decoupled
port so the two repos stay independent.
"""
import mlx.core as mx
import mlx.nn as nn

# ============================================================================
# Metal kernel: W8A8 GEMM with scale epilogue
# (ported from phase5_int8linear.py, unchanged numerics)
# ============================================================================
_HEADER = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

SM = SN = SK = 32
TM = TN = 2          # 2x2 fragment grid -> 64x64 per simdgroup
THREADS = 32          # 1 simdgroup per threadgroup


def _make_kernel(name):
    decls, zeros, loads_a, loads_b, runs, stores = [], [], [], [], [], []
    for i in range(TM):
        decls.append(f"auto ct_a{i} = op.get_left_input_cooperative_tensor<int8_t, int8_t, int>();")
        loads_a.append(f"""
      tensor<device int8_t, extents<int32_t, SK, SM>, tensor_inline> sA{i}(
          (device int8_t*)(A + (size_t)(row + {i} * SM) * K + k),
          extents<int32_t, SK, SM>(), array<int32_t, 2>{{1, K}});
      ct_a{i}.load(sA{i});""")
    for j in range(TN):
        decls.append(f"auto ct_b{j} = op.get_right_input_cooperative_tensor<int8_t, int8_t, int>();")
        # story 2.9: B is the (N, K) int8 weight, read transposed (was (K, N))
        loads_b.append(f"""
      tensor<device int8_t, extents<int32_t, SK, SN>, tensor_inline> sB{j}(
          (device int8_t*)(B + (size_t)(col + {j} * SN) * K + k),
          extents<int32_t, SK, SN>(), array<int32_t, 2>{{1, K}});
      ct_b{j}.load(sB{j});""")
    for i in range(TM):
        for j in range(TN):
            decls.append(
                f"auto ct_c{i}{j} = op.get_destination_cooperative_tensor<"
                f"decltype(ct_a{i}), decltype(ct_b{j}), int>();")
            zeros.append(
                f"for (ushort i = 0; i < ct_c{i}{j}.get_capacity(); ++i) ct_c{i}{j}.set(i, 0);")
            runs.append(f"op.run(ct_a{i}, ct_b{j}, ct_c{i}{j});")
            stores.append(f"""
    for (ushort i = 0; i < ct_c{i}{j}.get_capacity(); ++i) {{
        auto mi = ct_c{i}{j}.get_multidimensional_index(i);
        const int r = row + {i} * SM + mi[1];
        const int c = col + {j} * SN + mi[0];
        D[(size_t)r * N + c] = (half)((float)ct_c{i}{j}[i] * a_scale[r] * w_scale[c]);
    }}""")

    src = f"""
    constexpr int SM = {SM}, SN = {SN}, SK = {SK};

    const int M = int(dims[0]);
    const int N = int(dims[1]);
    const int K = int(dims[2]);
    (void)M;

    const int tgx = int(threadgroup_position_in_grid.x);
    const int tgy = int(threadgroup_position_in_grid.y);
    const int row = tgx * SM * {TM};
    const int col = tgy * SN * {TN};

    constexpr auto desc = matmul2d_descriptor(
        SM, SN, SK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    {chr(10).join('    ' + d for d in decls)}

    {chr(10).join('    ' + z for z in zeros)}

    for (int k = 0; k < K; k += SK) {{
      {''.join(loads_a)}
      {''.join(loads_b)}
      {chr(10).join('      ' + r for r in runs)}
    }}
    {''.join(stores)}
    """
    return mx.fast.metal_kernel(
        name=name,
        input_names=["A", "B", "a_scale", "w_scale", "dims"],
        output_names=["D"],
        header=_HEADER, source=src,
    )


_KERNEL = _make_kernel("int8_linear_epilogue_w8a8opt_nk")   # distinct cache name from int8-mlx
ROW_ALIGN = SM * TM  # 64


def _run_int8_linear(A, B, a_scale, w_scale, M, N, K):
    grid = ((M // (SM * TM)) * THREADS, N // (SN * TN), 1)
    (D,) = _KERNEL(
        inputs=[A, B, a_scale, w_scale, mx.array([M, N, K], mx.uint32)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
        grid=grid, threadgroup=(THREADS, 1, 1),
    )
    return D


# ============================================================================
# Metal kernel: fused per-row dynamic int8 activation quant
# (ported from fused_rowquant.py, unchanged numerics)
# ============================================================================
_ROWQUANT_SRC = """
    constexpr int NT = 256;
    const int K = int(dims[0]);
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32u;
    const uint sg = tid / 32u;

    const device half* xr = x + (size_t)row * K;
    device int8_t* cr = codes + (size_t)row * K;

    threadgroup float smax[NT / 32];

    float m = 1e-8f;
    for (int i = int(tid); i < K; i += NT) {
        m = max(m, fabs((float)xr[i]));
    }
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
    const float s = amax / 127.0f;
    const float inv = 127.0f / amax;
    if (tid == 0) scales[row] = s;

    for (int i = int(tid); i < K; i += NT) {
        cr[i] = (int8_t)clamp(rint((float)xr[i] * inv), -127.0f, 127.0f);
    }
"""

_ROWQUANT_KERNEL = mx.fast.metal_kernel(
    name="fused_rowquant_int8_w8a8opt",
    input_names=["x", "dims"],
    output_names=["codes", "scales"],
    source=_ROWQUANT_SRC,
)


def _rowquant_int8(x):
    """x: (M, K) fp16/float. Returns (codes int8 (M,K), scales fp32 (M,))."""
    M, K = x.shape
    codes, scales = _ROWQUANT_KERNEL(
        inputs=[x, mx.array([K], mx.uint32)],
        output_shapes=[(M, K), (M,)],
        output_dtypes=[mx.int8, mx.float32],
        grid=(M * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return codes, scales



# ============================================================================
# Metal kernel: small-row GEMV over the int8 weights (story 2.9)
# y[m, n] = ws[n] * sum_k x[m, k] * wq[n, k]: x in its own dtype, fp32 accumulate,
# output in x's dtype. One simdgroup owns _GEMV_R output rows; each lane reads 16
# contiguous int8 of each row per step (a simdgroup covers 512 of K), then
# simd_sum. The same (N, K) bytes the GEMM reads, so no second copy.
# ============================================================================
GEMV_MAX_ROWS = 8     # calls of up to this many rows take the GEMV; 9..63 are padded
_GEMV_R = 2           # output rows per simdgroup
_GEMV_SGS = 4         # simdgroups per threadgroup

_GEMV_SRC = f"""
    constexpr int MMAX = {GEMV_MAX_ROWS}, R = {_GEMV_R}, SGS = {_GEMV_SGS};
    const int M = int(dims[0]);
    const int N = int(dims[1]);
    const int K = int(dims[2]);
    const int lane = int(thread_index_in_simdgroup);
    const int n0 = (int(threadgroup_position_in_grid.x) * SGS
                    + int(simdgroup_index_in_threadgroup)) * R;
    if (n0 >= N) return;            // whole simdgroup: n0 is per simdgroup

    float acc[MMAX][R];
    for (int m = 0; m < MMAX; ++m) for (int r = 0; r < R; ++r) acc[m][r] = 0.0f;
    for (int k = lane * 16; k < K; k += 512) {{
      float4 w[R][4];
      for (int r = 0; r < R; ++r) {{
        const device char4* wp = (const device char4*)(W + (size_t)(n0 + r) * K + k);
        for (int i = 0; i < 4; ++i) w[r][i] = float4(wp[i]);
      }}
      for (int m = 0; m < M; ++m) {{
        const device auto* xp = x + (size_t)m * K + k;
        for (int i = 0; i < 4; ++i) {{
          const float4 xv = float4(float(xp[4 * i]), float(xp[4 * i + 1]),
                                   float(xp[4 * i + 2]), float(xp[4 * i + 3]));
          for (int r = 0; r < R; ++r) acc[m][r] += dot(xv, w[r][i]);
        }}
      }}
    }}
    for (int m = 0; m < M; ++m) {{
      for (int r = 0; r < R; ++r) {{
        const float t = simd_sum(acc[m][r]);
        if (lane == 0) D[(size_t)m * N + n0 + r] = static_cast<T>(t * ws[n0 + r]);
      }}
    }}
"""

_GEMV_KERNEL = mx.fast.metal_kernel(
    name="mlx_classifiers_int8_gemv_nk",
    input_names=["x", "W", "ws", "dims"],
    output_names=["D"],
    source=_GEMV_SRC,
)


def _gemv_int8(x, wq, ws, N, K):
    """x (M <= GEMV_MAX_ROWS, K) -> (M, N) in x's dtype, on the (N, K) int8 weights.
    Needs N % (_GEMV_R * _GEMV_SGS) == 0 and K % 16 == 0 (`_eligible` implies both)."""
    M = x.shape[0]
    (d,) = _GEMV_KERNEL(
        inputs=[x, wq, ws, mx.array([M, N, K], mx.uint32)],
        template=[("T", x.dtype)],
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
        grid=(N // _GEMV_R * 32, 1, 1),
        threadgroup=(32 * _GEMV_SGS, 1, 1),
    )
    return d

# ============================================================================
# W8A8 linear wrapper
# ============================================================================
class Int8LinearW8A8(nn.Module):
    """W8A8 wrapper: fast int8 GEMM path for prefill-shaped inputs; up to
    GEMV_MAX_ROWS rows run a GEMV on the same int8 weights (story 2.9). The
    stock linear is not kept: only build this from an `_eligible` linear.
    """

    def __init__(self, orig):
        super().__init__()
        # dequantize QAT weight -> fp32 (N, K), then per-output-channel
        # (per row of N) symmetric int8, kept (N, K) (story 2.9: the GEMM reads it
        # transposed and the small-row GEMV reads its rows; was pre-transposed).
        # Story 2.9: the same fp32 ops in mx on the GPU instead of numpy (bit-equal
        # int8 weights and scales on all 205 E2B linears; no host-side fp32 copies).
        wf = mx.dequantize(orig.weight, orig.scales, orig.biases,
                           group_size=orig.group_size, bits=orig.bits).astype(mx.float32)
        ws = mx.maximum(mx.abs(wf).max(axis=1), 1e-8) / 127.0       # (N,) fp32
        wq = mx.clip(mx.round(wf / ws[:, None]), -127, 127).astype(mx.int8)
        mx.eval(wq, ws)
        self._wq = wq                                        # (N, K) int8
        self._ws = ws                                        # (N,)
        self._N, self._K = wf.shape

    def __call__(self, x):
        orig_shape = x.shape
        xm = x.reshape(-1, self._K)
        M = xm.shape[0]
        if M <= GEMV_MAX_ROWS:
            out = _gemv_int8(xm, self._wq, self._ws, self._N, self._K)
            return out.reshape(*orig_shape[:-1], self._N)
        pad = (-M) % ROW_ALIGN
        if pad:
            xm = mx.concatenate([xm, mx.zeros((pad, self._K), dtype=xm.dtype)])
        Mp = xm.shape[0]
        # dynamic per-token symmetric int8 (fused single-pass kernel)
        codes, s = _rowquant_int8(xm.astype(mx.float16))
        out = _run_int8_linear(codes, self._wq, s, self._ws, Mp, self._N, self._K)
        if pad:
            out = out[:M]
        # story 5.21 fix: cast fp16 kernel output back to the INPUT's dtype
        # before returning -- otherwise a bf16 residual stream gets silently
        # promoted to fp32 for the rest of the model (the "+1160ms" tax).
        return out.reshape(*orig_shape[:-1], self._N).astype(x.dtype)


# ============================================================================
# Patch entry points (semantics match int8-mlx's patch_mlps / patch_projs)
# ============================================================================
def _decoder_layers(model):
    """The decoder layer list: ``model.language_model.model.layers`` on the gemma4
    multimodal wrapper (what mlx-lm loads for E2B), ``model.model.layers`` otherwise."""
    return getattr(model, "language_model", model).model.layers


def _eligible(lin):
    """A quantized linear without bias whose dims the kernels take (K % 32, N % 64);
    anything else stays stock (story 2.9: the wrapper has no stock fallback)."""
    if not isinstance(lin, nn.QuantizedLinear) or "bias" in lin:
        return False
    if getattr(lin, "mode", "affine") != "affine":
        return False
    N, K = lin.weight.shape[0], lin.weight.shape[1] * 32 // lin.bits
    return K % 32 == 0 and N % 64 == 0


def patch_mlps(model):
    """Replace every decoder layer's dense mlp.{gate,up,down}_proj with the
    W8A8 wrapper. Experts (MoE) are left untouched (stock 4-bit QAT)."""
    n = 0
    for layer in _decoder_layers(model):
        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(mlp, name)
            if _eligible(orig):
                setattr(mlp, name, Int8LinearW8A8(orig))
                n += 1
    return n


def patch_projs(model):
    """Replace every decoder layer's attention q/k/v/o projections with the
    W8A8 wrapper. kv-shared layers (no local k_proj/v_proj) are skipped for
    those two projections via hasattr, matching agent_91_wire.patch_projs."""
    n = 0
    for layer in _decoder_layers(model):
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if hasattr(attn, name) and _eligible(getattr(attn, name)):
                setattr(attn, name, Int8LinearW8A8(getattr(attn, name)))
                n += 1
    return n


def patch_model(model):
    """Apply :func:`patch_mlps` then :func:`patch_projs`; return ``(n_mlp, n_proj)``.
    The replaced stock linears are unreferenced once swapped; their buffers are
    returned to the OS here rather than left in MLX's cache (story 2.9)."""
    out = patch_mlps(model), patch_projs(model)
    mx.clear_cache()
    return out
