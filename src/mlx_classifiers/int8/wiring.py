# Vendored for mlx-lm-as-classifiers story 2.3 on 2026-10-16 from
#   /Users/maddie/Documents/dev/2026/aug/poorly-ported-gemma4-int8-optimisations/w8a8_wiring.py
#   (mtime 2026-08-09, sha256 7f7be1111c23a34704666935582b93e822f9dbef5fcd93e1bf746f06963159e2)
# Kernel sources and numerics are unchanged. Local changes, all in the patch
# entry points at the bottom: the layer walk goes through `_decoder_layers`
# (accepts the gemma4 multimodal wrapper or a bare text model), and
# `patch_model` applies both patches. Run on Gemma 4 E2B (35 layers; layers
# 15..34 are KV-shared and have no k_proj/v_proj, which `patch_projs` already
# skips via hasattr). Needs Apple M5 / Metal 4 (MPP matmul2d cooperative tensors).
# ruff: noqa
# Kept byte-for-byte with the source below this header, so formatting is off.
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
import numpy as np
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
        loads_b.append(f"""
      tensor<device int8_t, extents<int32_t, SN, SK>, tensor_inline> sB{j}(
          (device int8_t*)(B + (size_t)k * N + col + {j} * SN),
          extents<int32_t, SN, SK>(), array<int32_t, 2>{{1, N}});
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
        SM, SN, SK, false, false, true,
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


_KERNEL = _make_kernel("int8_linear_epilogue_w8a8opt")   # distinct cache name from int8-mlx
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
# W8A8 linear wrapper
# ============================================================================
class Int8LinearW8A8(nn.Module):
    """W8A8 wrapper: fast int8 GEMM path for prefill-shaped inputs,
    falls back to the original (stock-quantized) linear otherwise.
    """

    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        # dequantize QAT weight -> fp32 (N, K), then per-output-channel
        # (per row of N) symmetric int8, pre-transposed to (K, N).
        w = mx.dequantize(orig.weight, orig.scales, orig.biases,
                          group_size=orig.group_size, bits=orig.bits)
        wf = np.array(w.astype(mx.float32))
        amax = np.maximum(np.abs(wf).max(axis=1), 1e-8)
        ws = (amax / 127.0).astype(np.float32)              # (N,)
        wq = np.clip(np.round(wf / ws[:, None]), -127, 127).astype(np.int8)
        self._wq_t = mx.array(np.ascontiguousarray(wq.T))   # (K, N) int8
        self._ws = mx.array(ws)                              # (N,)
        self._N, self._K = wf.shape

    def __call__(self, x):
        orig_shape = x.shape
        xm = x.reshape(-1, self._K)
        M = xm.shape[0]
        if M < ROW_ALIGN or (self._K % 32) or (self._N % 64):
            return self.orig(x)
        pad = (-M) % ROW_ALIGN
        if pad:
            xm = mx.concatenate([xm, mx.zeros((pad, self._K), dtype=xm.dtype)])
        Mp = xm.shape[0]
        # dynamic per-token symmetric int8 (fused single-pass kernel)
        codes, s = _rowquant_int8(xm.astype(mx.float16))
        out = _run_int8_linear(codes, self._wq_t, s, self._ws, Mp, self._N, self._K)
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


def patch_mlps(model):
    """Replace every decoder layer's dense mlp.{gate,up,down}_proj with the
    W8A8 wrapper. Experts (MoE) are left untouched (stock 4-bit QAT)."""
    n = 0
    for layer in _decoder_layers(model):
        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(mlp, name)
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
            if hasattr(attn, name):
                setattr(attn, name, Int8LinearW8A8(getattr(attn, name)))
                n += 1
    return n


def patch_model(model):
    """Apply :func:`patch_mlps` then :func:`patch_projs`; return ``(n_mlp, n_proj)``."""
    return patch_mlps(model), patch_projs(model)
