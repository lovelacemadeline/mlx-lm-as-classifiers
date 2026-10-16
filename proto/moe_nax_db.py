"""Story 3.6: double-buffered per-expert NAX kernel (weight block k+1 loads while block k multiplies)."""
import sys
import mlx.core as mx
sys.path.insert(0, "proto")
import moe_nax
from moe_nax import bench

IMPL_DB = r"""
template <typename T, const int group_size, const int bits, const int BM, const int BK,
          const int BN, const int WM, const int WN>
METAL_FUNC void seg_db_impl(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, device T* y, threadgroup T* Ws, const int K, const int N,
    const int M, const int tile_y, const int tile_x, uint simd_gid, uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int WS_SIZE = BN * BK_padded;
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
  loader_w_t loader0(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  loader_w_t loader1(wl, scales, biases, K, Ws + WS_SIZE, simd_gid, simd_lid);
  loader1.next();
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
  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader0.load_unsafe();
    loader0.next(); loader0.next();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < K_it; k++) {
      const int cur = k & 1;
      if (k + 1 < K_it) {
        if (cur == 0) { loader1.load_unsafe(); loader1.next(); loader1.next(); }
        else { loader0.load_unsafe(); loader0.next(); loader0.next(); }
      }
      const threadgroup T* Wc = Ws + cur * WS_SIZE;
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
        Btile.template load<T, BK_padded, 1>(Wc + tn * BK_padded + kk1);
        tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
        (void)compiler_barrier;
      }
      x += BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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
SRC_DB = moe_nax.SRC.replace("threadgroup T Ws[BN * BK_padded];", "threadgroup T Ws[2 * BN * BK_padded];").replace("seg_qmm_t_nax_impl", "seg_db_impl")
SRC_DB = SRC_DB.replace("  for (int t = 0;", "  for (int t = 0;")  # tiles of one expert run back to back
old = moe_nax.IMPL
moe_nax.IMPL = old + IMPL_DB
K_DB = mx.fast.metal_kernel(name="seg_gather_qmm_nax_db", input_names=["x", "w", "scales", "biases", "offs"],
                            output_names=["y"], header=moe_nax.header(), source=SRC_DB)
moe_nax.IMPL = old


def run(k, xs, wq, s, b, offs, N, K, BM=16, BN=64, BK=64, WM=1, WN=2):
    tg = 32 * WM * WN
    return k(inputs=[xs, wq, s, b, offs], template=[("T", mx.bfloat16), ("GS", 64), ("BITS", 4), ("BM", BM), ("BN", BN), ("BK", BK), ("WM", WM), ("WN", WN), ("K", K), ("N", N)],
             grid=(tg * (N // BN), offs.shape[0] - 1, 1), threadgroup=(tg, 1, 1), output_shapes=[(xs.shape[0], N)], output_dtypes=[mx.bfloat16])[0]


if __name__ == "__main__":
    E = 128
    for (N, K) in [(704, 2816), (2816, 704)]:
        w = (mx.random.normal((E, N, K)) * 0.02).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, s, b); del w
        for L in [64, 97, 256, 512]:
            x = mx.random.normal((L, K)).astype(mx.bfloat16)
            idx = mx.random.randint(0, E, (L, 8))
            flat = idx.flatten(); order = mx.argsort(flat); xs = x[order // 8]; ids = flat[order]
            offs = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum((ids[:, None] == mx.arange(E)[None]).sum(0).astype(mx.int32))])
            mx.eval(xs, ids, offs)
            stock = lambda: mx.gather_qmm(xs[:, None], wq, s, b, rhs_indices=ids, transpose=True, group_size=64, bits=4, sorted_indices=True)[:, 0]
            ref = stock()
            out = [f"N={N} L={L:4d} stock {bench(stock):5.2f}  single {bench(lambda: moe_nax.seg_qmm(xs, wq, s, b, ids)):5.2f}"]
            for cfg in [(16, 64, 64, 1, 2), (16, 32, 64, 1, 1), (16, 64, 32, 1, 2), (16, 128, 64, 1, 4), (32, 64, 64, 2, 2)]:
                if N % cfg[1]:
                    continue
                f = lambda: run(K_DB, xs, wq, s, b, offs, N, K, *cfg)
                out.append(f"db{cfg[:3]} {bench(f):5.2f}{'' if mx.array_equal(ref, f()).item() else '!'}")
            print("  ".join(out), flush=True)
