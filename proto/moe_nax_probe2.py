"""Story 3.6: loader-only cost with the loads kept alive."""
import sys
import mlx.core as mx
sys.path.insert(0, "proto")
import moe_nax
from moe_nax import bench

impl = moe_nax.IMPL.replace("seg_qmm_t_nax_impl", "probe_impl")
impl = impl.replace("tile_matmad_nax(Dtile", "if (0) tile_matmad_nax(Dtile")
impl = impl.replace("      x += BK;\n      loader_w.next();", "      acc_keep += float(Ws[simd_lid]);\n      x += BK;\n      loader_w.next();")
impl = impl.replace("  dispatch_bool(!is_unaligned_sm", "  float acc_keep = 0.0f;\n  dispatch_bool(!is_unaligned_sm")
impl = impl.replace("    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (sgp_sm > 0) {", "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (acc_keep == 12345.0f) y[simd_lid] = T(acc_keep);\n    if (sgp_sm > 0) {")
assert "acc_keep += " in impl and "12345" in impl
src = moe_nax.SRC.replace("seg_qmm_t_nax_impl", "probe_impl")
old = moe_nax.IMPL; moe_nax.IMPL = impl; hdr = moe_nax.header(); moe_nax.IMPL = old
k = mx.fast.metal_kernel(name="probe_load_alive", input_names=["x", "w", "scales", "biases", "offs"], output_names=["y"], header=hdr, source=src)

E, N, K, L = 128, 704, 2816, 97
w = (mx.random.normal((E, N, K)) * 0.02).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, s, b); del w
x = mx.random.normal((L, K)).astype(mx.bfloat16)
idx = mx.random.randint(0, E, (L, 8))
flat = idx.flatten(); order = mx.argsort(flat); xs = x[order // 8]; ids = flat[order]
offs = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum((ids[:, None] == mx.arange(E)[None]).sum(0).astype(mx.int32))])
mx.eval(xs, ids, offs)
f = lambda: k(inputs=[xs, wq, s, b, offs], template=[("T", mx.bfloat16), ("GS", 64), ("BITS", 4), ("BM", 16), ("BN", 64), ("BK", 64), ("WM", 1), ("WN", 2), ("K", K), ("N", N)],
              grid=(64 * (N // 64), E, 1), threadgroup=(64, 1, 1), output_shapes=[(xs.shape[0], N)], output_dtypes=[mx.bfloat16])[0]
print(f"load only (alive) {bench(f):.2f} ms;  full {bench(lambda: moe_nax.seg_qmm(xs, wq, s, b, ids)):.2f} ms")
