"""Per-expert gather-QMM for MoE prefill chunks of tens to hundreds of tokens (story 3.6).

Why: a delta-cached decision on Gemma 4 26B-A4B prefills about 100 tokens. With
top-8 routing over 128 experts that is about 800 (token, expert) rows, about 6
per expert, so every expert is touched. MLX 0.32 runs those rows through
``affine_gather_qmm_rhs_nax`` (``mx.gather_qmm`` with ``sorted_indices``) in
64-row tiles, and a tile that spans k experts runs k full 64-row matmuls: about
ten times the useful work. Story 3.6 measured the expert matmuls at 389 of the
546 ms of a 97-token cache-only prefill on the 26B.

What: :func:`segmented_gather_qmm` gives each threadgroup one expert and runs
16-row tiles over that expert's rows only. The tile body is MLX's own
``qmm_t_nax_tgp_impl`` (weights dequantised by MLX's ``QuantizedBlockLoader``,
NAX ``tile_matmad_nax``, K walked in the same order), so the output is
**bit-identical** to ``mx.gather_qmm``'s NAX path. The kernel header is built
at first use from the ``quantized_nax.h`` / ``steel/gemm/nax.h`` headers that
ship inside the installed ``mlx`` wheel, so it tracks the installed version;
:func:`available` refuses anything but MLX 0.32 on a NAX GPU (Apple M5,
``applegpu_g17``), and :func:`patch_model` checks bit-identity against
``mx.gather_qmm`` once before patching anything.

When: MLX only takes the rhs path for ``rows >= 4 * experts`` (below that it runs
a per-row kernel with different rounding), and past one 16-row tile per expert
its 64-row tiles stop wasting work, so :func:`use_segmented` engages for
``4 * experts <= rows <= 16 * experts``: 64 to 256 tokens on the 26B. Outside
that window, and for the L=1 readout pass, the stock path runs unchanged.

``patch_model`` swaps the class of every eligible ``SwitchGLU`` (4-bit affine,
group 64, no bias) to :class:`SegmentedSwitchGLU`; dense models have none and
are left alone. ``MLXRunner.load(moe_kernel=False)`` skips it. Importing this
module imports ``mlx`` and ``mlx_lm``, so the runner imports it lazily.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

__all__ = [
    "MAX_ROWS_PER_EXPERT",
    "MIN_ROWS_PER_EXPERT",
    "SegmentedSwitchGLU",
    "available",
    "expert_offsets",
    "patch_model",
    "segmented_gather_qmm",
    "use_segmented",
]

#: MLX 0.32 ``gather_qmm`` takes its rhs (NAX) path from this many rows per expert.
MIN_ROWS_PER_EXPERT = 4
#: Past one 16-row tile per expert on average the stock 64-row tiles win.
MAX_ROWS_PER_EXPERT = 16

_BM, _BN, _BK, _WM, _WN = 16, 64, 64, 1, 2
_GROUP_SIZE, _BITS = 64, 4


def use_segmented(rows: int, experts: int) -> bool:
    """Whether ``rows`` sorted (token, expert) rows over ``experts`` take this kernel."""
    return MIN_ROWS_PER_EXPERT * experts <= rows <= MAX_ROWS_PER_EXPERT * experts


def _include_root() -> Path:
    return Path(mx.__file__).parent / "include"


def available() -> bool:
    """MLX 0.32 on a NAX GPU (Apple M5 and later), with its kernel headers installed."""
    if not mx.__version__.startswith("0.32."):
        return False
    try:
        arch = mx.device_info().get("architecture", "")
    except Exception:  # noqa: BLE001 - no Metal device
        return False
    m = re.match(r"applegpu_g(\d+)", str(arch))
    kernels = _include_root() / "mlx/backend/metal/kernels"
    return bool(m and int(m.group(1)) >= 17 and (kernels / "quantized_nax.h").exists())


def _inline(rel: str, seen: set[str]) -> str:
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in (_include_root() / rel).read_text().splitlines():
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(_inline(m.group(1), seen))
        elif line.strip() != "#pragma once":
            out.append(line)
    return "\n".join(out)


# MLX 0.32's qmm_t_nax_tgp_impl (quantized_nax.h), aligned N and K, with K/N/M by
# value (M is the expert's row count) and the tile position passed in.
_IMPL = r"""
template <typename T, const int group_size, const int bits, const int BM, const int BK,
          const int BN, const int WM, const int WN>
METAL_FUNC void seg_qmm_t_nax_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Ws, const int K, const int N,
    const int M, const int tile_y, const int tile_x, uint simd_gid, uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tile_y * BM;
  const int y_col = tile_x * BN;
  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);
  const short sgp_sm = max(0, min(int(SM), M - (y_row + tm)));
  const bool is_unaligned_sm = (sgp_sm != SM);
  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();
  x += tm * K;
  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    for (int k = 0; k < K; k += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < BK; kk1 += SK) {
        NAXTile<T, TM, TK> Atile;
        NAXTile<T, TN, TK> Btile;
        volatile int compiler_barrier;
        if constexpr (kAlignedM.value) {
          Atile.load(x + kk1, K);
        } else {
          Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
        }
        Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
        tile_matmad_nax(
            Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
        (void)compiler_barrier;
      }
      x += BK;
      loader_w.next();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgp_sm > 0) {
      if constexpr (kAlignedM.value) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(SN, sgp_sm));
      }
    }
  });
}
"""

# One threadgroup per (column block, expert); an expert with no rows exits at once.
_SOURCE = r"""
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];
  const int e = int(threadgroup_position_in_grid.y);
  const int r0 = offs[e];
  const int M = offs[e + 1] - r0;
  if (M == 0) return;
  const size_t w_stride = size_t(N) * size_t(K) * BITS / 32;
  const size_t s_stride = size_t(N) * size_t(K / GS);
  for (int t = 0; t * BM < M; t++) {
    seg_qmm_t_nax_impl<T, GS, BITS, BM, BK, BN, WM, WN>(
        w + e * w_stride, scales + e * s_stride, biases + e * s_stride,
        x + size_t(r0) * K, y + size_t(r0) * N, Ws, K, N, M, t,
        int(threadgroup_position_in_grid.x), simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
  }
"""


def _header() -> str:
    # metal_kernel's preamble already includes kernels/utils.h and what it includes.
    k = "mlx/backend/metal/kernels/"
    seen = {k + f for f in ("utils.h", "bf16.h", "bf16_math.h", "complex.h", "defines.h")}
    seen.add(k + "logging.h")
    parts = [_inline(k + "steel/gemm/nax.h", seen)]
    q = (_include_root() / (k + "quantized_nax.h")).read_text().splitlines()
    # helpers and loaders only: stop before the stock kernels (adjust_matrix_offsets on)
    stop = next(i for i, ln in enumerate(q) if "METAL_FUNC void adjust_matrix_offsets" in ln)
    stop = max(i for i in range(stop) if q[i].startswith("template"))
    parts.append(
        "\n".join(
            ln for ln in q[:stop] if "function_constant" not in ln and not ln.startswith("#include")
        )
    )
    parts.append(_IMPL)
    return "\n".join(parts)


@cache
def _kernel() -> Any:
    return mx.fast.metal_kernel(
        name="mlx_classifiers_segmented_gather_qmm",
        input_names=["x", "w", "scales", "biases", "offs"],
        output_names=["y"],
        header=_header(),
        source=_SOURCE,
    )


def expert_offsets(sorted_ids: Any, experts: int) -> Any:
    """``(experts + 1,)`` int32 row offsets of each expert in ``sorted_ids``."""
    counts = (sorted_ids[:, None] == mx.arange(experts)[None]).sum(0).astype(mx.int32)
    return mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(counts)])


def segmented_gather_qmm(x: Any, weight: Any, scales: Any, biases: Any, offs: Any) -> Any:
    """``x`` (rows, K) sorted by expert, ``offs`` from :func:`expert_offsets`, ``weight``
    (E, N, K/8) 4-bit affine group 64 -> (rows, N); equals ``mx.gather_qmm(x[:, None],
    ..., transpose=True, sorted_indices=True)[:, 0]`` bit for bit."""
    experts, n, _ = weight.shape
    k = x.shape[-1]
    threads = 32 * _WM * _WN
    (y,) = _kernel()(
        inputs=[x, weight, scales, biases, offs],
        template=[
            ("T", x.dtype),
            ("GS", _GROUP_SIZE),
            ("BITS", _BITS),
            ("BM", _BM),
            ("BN", _BN),
            ("BK", _BK),
            ("WM", _WM),
            ("WN", _WN),
            ("K", k),
            ("N", n),
        ],
        grid=(threads * (n // _BN), experts, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(x.shape[0], n)],
        output_dtypes=[x.dtype],
    )
    return y


def _eligible(lin: Any) -> bool:
    return (
        isinstance(lin, QuantizedSwitchLinear)
        and getattr(lin, "mode", "affine") == "affine"
        and lin.bits == _BITS
        and lin.group_size == _GROUP_SIZE
        and lin.get("biases") is not None
        and "bias" not in lin
        and lin.weight.shape[1] % _BN == 0
        and lin.input_dims % _BK == 0
    )


def _project(lin: Any, x: Any, offs: Any) -> Any:
    return segmented_gather_qmm(x, lin["weight"], lin["scales"], lin["biases"], offs)


class SegmentedSwitchGLU(SwitchGLU):
    """``SwitchGLU`` whose sorted path runs :func:`segmented_gather_qmm` inside the
    :func:`use_segmented` window (same values, bit for bit); stock elsewhere."""

    def __call__(self, x: Any, indices: Any) -> Any:
        experts = self.up_proj.num_experts
        if self.training or not use_segmented(indices.size, experts):
            return super().__call__(x, indices)
        xs, idx, inv_order = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
        xs = xs.reshape(xs.shape[0], xs.shape[-1])
        offs = expert_offsets(idx, experts)
        up = _project(self.up_proj, xs, offs)
        gate = _project(self.gate_proj, xs, offs)
        out = _project(self.down_proj, self.activation(up, gate), offs)
        return _scatter_unsort(out[:, None], inv_order, indices.shape).squeeze(-2)


@cache
def _self_check() -> bool:
    """One small problem through both paths: bit-identical or the patch is refused."""
    experts, n, k, rows = 8, 128, 256, 8 * 6
    w = (mx.random.normal((experts, n, k), key=mx.random.key(3)) * 0.05).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=_GROUP_SIZE, bits=_BITS)
    x = mx.random.normal((rows, k), key=mx.random.key(4)).astype(mx.bfloat16)
    ids = mx.sort(mx.random.randint(0, experts, (rows,), key=mx.random.key(5))).astype(mx.uint32)
    ref = mx.gather_qmm(
        x[:, None], wq, s, b, rhs_indices=ids, transpose=True,
        group_size=_GROUP_SIZE, bits=_BITS, sorted_indices=True,
    )[:, 0]  # fmt: skip
    got = segmented_gather_qmm(x, wq, s, b, expert_offsets(ids, experts))
    return bool(mx.array_equal(ref, got).item())


def patch_model(model: Any) -> int:
    """Swap every eligible ``SwitchGLU`` in ``model`` to :class:`SegmentedSwitchGLU`.
    Returns how many were swapped (0 for dense models, an unsupported MLX or GPU,
    or a failed bit-identity self-check)."""
    targets = [
        m
        for _, m in model.named_modules()
        if type(m) is SwitchGLU and all(_eligible(p) for p in (m.up_proj, m.gate_proj, m.down_proj))
    ]
    if not targets or not available() or not _self_check():
        return 0
    for m in targets:
        m.__class__ = SegmentedSwitchGLU
    return len(targets)
