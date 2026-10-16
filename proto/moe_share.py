"""Story 3.6: how much of a 97-token cache-only prefill on the 26B is the expert matmuls?

Times the delta prefill with the stock SwitchGLU, then with SwitchGLU stubbed
(zeros of the right shape; numerics meaningless, cost of everything else)."""
import copy, statistics, sys, time
import mlx.core as mx
from mlx_lm.models import switch_layers
from mlx_lm.models.cache import make_prompt_cache
from mlx_classifiers.mlx_runner import MLXRunner

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
    return statistics.median(ts)

stock = switch_layers.SwitchGLU.__call__
for L in (1, 97):
    a = run(L)
    switch_layers.SwitchGLU.__call__ = lambda self, x, idx: mx.zeros((*idx.shape, x.shape[-1]), x.dtype)
    b = run(L)
    switch_layers.SwitchGLU.__call__ = stock
    print(f"L={L}: stock {a:.1f} ms, experts stubbed {b:.1f} ms -> experts {a - b:.1f} ms ({(a - b) / a:.0%})")
