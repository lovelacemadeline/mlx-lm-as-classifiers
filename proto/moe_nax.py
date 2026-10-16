"""Story 3.6: per-expert NAX gather-QMM for sorted MoE rows (BM=16 tiles, one expert per threadgroup).

MLX 0.32's affine_gather_qmm_rhs_nax runs 64-row tiles, and a tile spanning k
experts runs k full 64-row matmuls. With ~6 rows per expert (a ~100-token
delta, top-8 of 128) that is ~10x padded work. This kernel gives each
threadgroup one expert and runs 16-row tiles over that expert's rows only.
Usage: .venv/bin/python proto/moe_nax.py"""
import re, statistics, time
from pathlib import Path
import mlx.core as mx

INC = Path(mx.__file__).parent / "include"


def inline(rel, seen):
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in (INC / rel).read_text().splitlines():
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(inline(m.group(1), seen))
        elif line.strip() == "#pragma once":
            continue
        else:
            out.append(line)
    return "\n".join(out)


def header():
    # metal_kernel's preamble already includes kernels/utils.h (and bf16, defines)
    seen = {"mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/bf16.h",
            "mlx/backend/metal/kernels/bf16_math.h", "mlx/backend/metal/kernels/complex.h",
            "mlx/backend/metal/kernels/defines.h", "mlx/backend/metal/kernels/logging.h"}
    parts = [inline("mlx/backend/metal/kernels/steel/gemm/nax.h", seen)]
    q = (INC / "mlx/backend/metal/kernels/quantized_nax.h").read_text().splitlines()
    # helpers only (loaders); stop before the first stock kernel impl, drop function constants
    end = next(i for i, l in enumerate(q) if "METAL_FUNC void adjust_matrix_offsets" in l)
    end = max(i for i in range(end) if q[i].startswith("template"))
    body = [l for l in q[:end] if "function_constant" not in l and not l.startswith("#include")]
    parts.append("\n".join(body))
    parts.append(IMPL)
    return "\n".join(parts)


IMPL = r"""
// qmm_t_nax_tgp_impl from MLX 0.32 quantized_nax.h, with K/N/M by value (M is per expert).
template <typename T, const int group_size, const int bits, const int BM, const int BK,
          const int BN, const int WM, const int WN>
METAL_FUNC void seg_qmm_t_nax_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Ws, const int K, const int N,
    const int M, const int tile_y, const int tile_x, uint simd_gid, uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
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
        tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
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

SRC = r"""
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];
  const int e = int(threadgroup_position_in_grid.y);
  const int r0 = offs[e];
  const int M = offs[e + 1] - r0;
  if (M == 0) return;
  const size_t w_stride = size_t(N) * size_t(K) * BITS / 32;   // uint32 words per expert
  const size_t s_stride = size_t(N) * size_t(K / GS);
  for (int t = 0; t * BM < M; t++) {
    seg_qmm_t_nax_impl<T, GS, BITS, BM, BK, BN, WM, WN>(
        w + e * w_stride, scales + e * s_stride, biases + e * s_stride,
        x + size_t(r0) * K, y + size_t(r0) * N, Ws, K, N, M, t,
        int(threadgroup_position_in_grid.x), simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
  }
"""

_KERNELS = {}


def kernel():
    if "k" not in _KERNELS:
        _KERNELS["k"] = mx.fast.metal_kernel(
            name="seg_gather_qmm_nax", input_names=["x", "w", "scales", "biases", "offs"],
            output_names=["y"], header=header(), source=SRC)
    return _KERNELS["k"]


def seg_qmm(x, w, s, b, ids, BM=16, BN=64, BK=64, WM=1, WN=2, bits=4, gs=64):
    """x (R, K) sorted by expert ids (R,); w (E, N, K*bits/32) affine -> (R, N)."""
    E, N, _ = w.shape
    K = x.shape[-1]
    counts = (ids[:, None] == mx.arange(E)[None]).sum(0).astype(mx.int32)
    offs = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(counts)])
    tg = 32 * WM * WN
    (y,) = kernel()(inputs=[x, w, s, b, offs],
                    template=[("T", x.dtype), ("GS", gs), ("BITS", bits), ("BM", BM), ("BN", BN), ("BK", BK),
                              ("WM", WM), ("WN", WN), ("K", K), ("N", N)],
                    grid=(tg * (N // BN), E, 1), threadgroup=(tg, 1, 1),
                    output_shapes=[(x.shape[0], N)], output_dtypes=[x.dtype])
    return y


def bench(make, reps=7, inner=20):
    mx.eval(make()); mx.synchronize(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval([make() for _ in range(inner)]); mx.synchronize()
        ts.append((time.perf_counter() - t) / inner)
    return statistics.median(ts) * 1e3


if __name__ == "__main__":
    E = 128
    for (N, K) in [(704, 2816), (2816, 704)]:
        w = (mx.random.normal((E, N, K)) * 0.02).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, s, b); del w
        for L in [8, 32, 97, 128, 256, 512, 1148]:
            x = mx.random.normal((L, K)).astype(mx.bfloat16)
            idx = mx.random.randint(0, E, (L, 8))
            flat = idx.flatten(); order = mx.argsort(flat); xs = x[order // 8]; ids = flat[order]
            mx.eval(xs, ids)
            stock = lambda: mx.gather_qmm(xs[:, None], wq, s, b, rhs_indices=ids, transpose=True, group_size=64, bits=4, sorted_indices=True)[:, 0]
            ref, got = stock(), seg_qmm(xs, wq, s, b, ids)
            d = mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))
            eq = mx.mean((ref == got).astype(mx.float32)).item()
            t1, t2 = bench(stock), bench(lambda: seg_qmm(xs, wq, s, b, ids))
            print(f"N={N} K={K} L={L:4d} stock {t1:6.2f} ms  seg {t2:6.2f} ms  x{t1 / t2:4.1f}  maxdiff {d.max().item():.1e}  bit-equal {eq:.2%}")
