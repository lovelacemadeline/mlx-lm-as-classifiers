"""Story 3.6: cache-only prefill cost vs delta length L, from a ~1046-token cache.

Usage: .venv/bin/python proto/delta_sweep.py [MODEL]
Separates fixed cost from per-token cost, and the sorted vs unsorted MoE path."""
import copy, statistics, sys, time
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_classifiers.mlx_runner import MLXRunner

model_id = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
r = MLXRunner.load(model_id, prompt_cache=None)
ids = mx.random.randint(1000, 200000, (1400,), key=mx.random.key(1)).tolist()
base = make_prompt_cache(r.model)
r._cache_only(ids[:1046], base)

def run(L, reps=5):
    ts = []
    for i in range(reps + 1):
        c = copy.deepcopy(base); mx.eval([x.state for x in c]); mx.synchronize()
        t = time.perf_counter()
        r._backbone(mx.array(ids[1046:1046 + L])[None], cache=c)
        mx.eval([x.state for x in c]); mx.synchronize()
        if i: ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)

for L in [1, 2, 4, 7, 8, 9, 16, 32, 64, 97, 128, 192, 256]:
    print(f"L={L:4d}  {run(L):7.1f} ms")
