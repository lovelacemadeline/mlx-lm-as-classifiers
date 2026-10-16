"""Story 3.6: the per-expert NAX kernel inside the 26B's SwitchGLU — delta prefill time and exactness."""
import copy, statistics, sys, time
import mlx.core as mx
sys.path.insert(0, "proto")
from moe_nax import kernel
from mlx_lm.models import switch_layers
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from mlx_lm.models.cache import make_prompt_cache
from mlx_classifiers.mlx_runner import MLXRunner

MIN_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 512


def proj(lin, x, offs, E):
    N, K = lin.weight.shape[1], x.shape[-1]
    (y,) = kernel()(inputs=[x, lin.weight, lin.scales, lin.biases, offs],
                    template=[("T", x.dtype), ("GS", lin.group_size), ("BITS", lin.bits), ("BM", 16), ("BN", 64), ("BK", 64),
                              ("WM", 1), ("WN", 2), ("K", K), ("N", N)],
                    grid=(64 * (N // 64), E, 1), threadgroup=(64, 1, 1),
                    output_shapes=[(x.shape[0], N)], output_dtypes=[x.dtype])
    return y


stock_call = switch_layers.SwitchGLU.__call__


def patched(self, x, indices):
    if indices.size < MIN_ROWS:
        return stock_call(self, x, indices)
    E = self.up_proj.weight.shape[0]
    xe = mx.expand_dims(x, (-2, -3))
    xs, idx, inv = _gather_sort(xe, indices)
    xs = xs[:, 0]
    offs = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum((idx[:, None] == mx.arange(E)[None]).sum(0).astype(mx.int32))])
    up = proj(self.up_proj, xs, offs, E)
    gate = proj(self.gate_proj, xs, offs, E)
    h = proj(self.down_proj, self.activation(up, gate), offs, E)
    return _scatter_unsort(h[:, None], inv, indices.shape).squeeze(-2)


r = MLXRunner.load("mlx-community/gemma-4-26B-A4B-it-qat-4bit", prompt_cache=None)
ids = mx.random.randint(1000, 200000, (1400,), key=mx.random.key(1)).tolist()
base = make_prompt_cache(r.model)
r._cache_only(ids[:1046], base)


def run(L, reps=7):
    ts = []
    for i in range(reps + 1):
        c = copy.deepcopy(base); mx.eval([x.state for x in c]); mx.synchronize()
        t = time.perf_counter()
        r._backbone(mx.array(ids[1046:1046 + L])[None], cache=c)
        mx.eval([x.state for x in c]); mx.synchronize()
        if i: ts.append((time.perf_counter() - t) * 1e3)
    h = r._backbone(mx.array(ids[1046 + L:1047 + L])[None], cache=c)
    return statistics.median(ts), h


for L in (64, 97, 128, 256, 1148 - 1046 + 0):
    a, ha = run(L)
    switch_layers.SwitchGLU.__call__ = patched
    b, hb = run(L)
    switch_layers.SwitchGLU.__call__ = stock_call
    print(f"L={L}: stock {a:.1f} ms, per-expert NAX {b:.1f} ms  (x{a / b:.2f})  next hidden identical: {mx.array_equal(ha, hb).item()}", flush=True)
