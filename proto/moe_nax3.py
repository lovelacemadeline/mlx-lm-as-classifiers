"""Story 3.6: per-expert NAX kernel v3 — one weight row per thread, 16-byte loads, next block prefetched
into registers while the current block multiplies. 4-bit, group 64, BK = 64 (one group per row per block)."""
import sys
import mlx.core as mx
sys.path.insert(0, "proto")
import moe_nax
from moe_nax import bench

IMPL3 = r"""
template <typename T>
METAL_FUNC void deq_row64(uint4 a, uint4 b, T scale, T bias, threadgroup T* dst) {
  // Bit-for-bit MLX's dequantize<T, 2, 4> over 32 bytes: low nibble then high nibble.
  T s[2] = {scale, scale / static_cast<T>(16.0f)};
  uint words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  STEEL_PRAGMA_UNROLL
  for (int wi = 0; wi < 8; wi++) {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 4; j++) {
      const uint8_t byte = uint8_t((words[wi] >> (8 * j)) & 0xffu);
      dst[8 * wi + 2 * j] = s[0] * (byte & 0x0f) + bias;
      dst[8 * wi + 2 * j + 1] = s[1] * (byte & 0xf0) + bias;
    }
  }
}

template <typename T, const int BM, const int BN, const int WM, const int WN>
METAL_FUNC void seg3_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Ws, const int K, const int N,
    const int M, const int tile_y, const int tile_x, uint simd_gid, uint simd_lid) {
  constexpr int BK = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  static_assert(BN == WM * WN * 32, "one weight row per thread");
  const int K_w4 = K / 32;          // uint4 per weight row (32 nibbles... 4-bit: 8 per uint32, 32 per uint4)
  const int K_g = K / 64;
  const int y_row = tile_y * BM;
  const int y_col = tile_x * BN;
  const int t = int(simd_gid) * 32 + int(simd_lid);
  const device uint4* wrow = (const device uint4*)w + size_t(y_col + t) * K_w4;
  const device T* srow = scales + size_t(y_col + t) * K_g;
  const device T* brow = biases + size_t(y_col + t) * K_g;
  threadgroup T* dst = Ws + t * BK_padded;
  x += y_row * static_cast<int64_t>(K);
  y += y_row * static_cast<int64_t>(N) + y_col;
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
  const int K_it = K / BK;
  uint4 a = wrow[0], b = wrow[1];
  T sc = srow[0], bi = brow[0];
  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      deq_row64<T>(a, b, sc, bi, dst);
      if (k + 1 < K_it) {
        a = wrow[2 * (k + 1)]; b = wrow[2 * (k + 1) + 1];
        sc = srow[k + 1]; bi = brow[k + 1];
      }
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
SRC3 = r"""
  constexpr int BK_padded = (64 + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];
  const int e = int(threadgroup_position_in_grid.y);
  const int r0 = offs[e];
  const int M = offs[e + 1] - r0;
  if (M == 0) return;
  const size_t w_stride = size_t(N) * size_t(K) / 8;
  const size_t s_stride = size_t(N) * size_t(K / 64);
  for (int t = 0; t * BM < M; t++) {
    seg3_impl<T, BM, BN, WM, WN>(
        w + e * w_stride, scales + e * s_stride, biases + e * s_stride,
        x + size_t(r0) * K, y + size_t(r0) * N, Ws, K, N, M, t,
        int(threadgroup_position_in_grid.x), simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
  }
"""
old = moe_nax.IMPL
moe_nax.IMPL = old + IMPL3
K3 = mx.fast.metal_kernel(name="seg_gather_qmm_nax_v3", input_names=["x", "w", "scales", "biases", "offs"],
                          output_names=["y"], header=moe_nax.header(), source=SRC3)
moe_nax.IMPL = old


def run3(xs, wq, s, b, offs, BM=16, BN=64, WM=1, WN=2):
    E, N, _ = wq.shape
    K = xs.shape[-1]
    tg = 32 * WM * WN
    return K3(inputs=[xs, wq, s, b, offs], template=[("T", xs.dtype), ("BM", BM), ("BN", BN), ("WM", WM), ("WN", WN), ("K", K), ("N", N)],
              grid=(tg * (N // BN), E, 1), threadgroup=(tg, 1, 1), output_shapes=[(xs.shape[0], N)], output_dtypes=[xs.dtype])[0]


if __name__ == "__main__":
    E = 128
    for (N, K) in [(704, 2816), (2816, 704)]:
        w = (mx.random.normal((E, N, K)) * 0.02).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, s, b); del w
        for L in [64, 97, 256, 512, 1148]:
            x = mx.random.normal((L, K)).astype(mx.bfloat16)
            idx = mx.random.randint(0, E, (L, 8))
            flat = idx.flatten(); order = mx.argsort(flat); xs = x[order // 8]; ids = flat[order]
            offs = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum((ids[:, None] == mx.arange(E)[None]).sum(0).astype(mx.int32))])
            mx.eval(xs, ids, offs)
            stock = lambda: mx.gather_qmm(xs[:, None], wq, s, b, rhs_indices=ids, transpose=True, group_size=64, bits=4, sorted_indices=True)[:, 0]
            ref = stock()
            out = [f"N={N} L={L:4d} stock {bench(stock):5.2f}  v1 {bench(lambda: moe_nax.seg_qmm(xs, wq, s, b, ids)):5.2f}"]
            for cfg in [(16, 64, 1, 2), (16, 128, 1, 4), (16, 32, 1, 1), (32, 64, 2, 1)]:
                if N % cfg[1]:
                    continue
                try:
                    f = lambda: run3(xs, wq, s, b, offs, *cfg)
                    out.append(f"v3{cfg[:2]} {bench(f):5.2f}{'' if mx.array_equal(ref, f()).item() else '!'}")
                except Exception as ex:
                    out.append(f"v3{cfg} ERR {str(ex)[:300]}")
            print("  ".join(out), flush=True)
