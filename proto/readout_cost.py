"""Story 2.5 prototype: what does a logit readout cost on Gemma 4 E2B?

Three ways to get the next-token distribution after an N-token prompt:
  naive : model(all N tokens) -> logits[:, -1]      (all 35 layers + lm_head over N rows)
  split : model(N-1 tokens) cache-only, then model(last 1 token) -> logits
          (this is what mlx_lm.generate_step does; MLX laziness should prune the
           20 KV-consumer layers and the lm_head for the N-1 chunk)
  prefix: as split, but the first P tokens come from a saved prefix cache
Prints ms per call and implied tok/s. Run: .venv/bin/python proto/readout_cost.py
"""
import copy, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL = "mlx-community/gemma-4-E2B-it-qat-4bit"
model, tok = load(MODEL)
V = model.args.text_config["vocab_size"] if hasattr(model.args, "text_config") else None

def t(fn, reps=3):
    fn(); mx.synchronize()
    best = 1e9
    for _ in range(reps):
        s = time.perf_counter(); fn(); mx.synchronize(); best = min(best, time.perf_counter() - s)
    return best * 1e3

def naive(ids):
    cache = make_prompt_cache(model)
    out = model(ids[None], cache=cache)[:, -1, :]
    mx.eval(out); return out

def split(ids, prefix_cache=None, n_prefix=0):
    cache = copy.deepcopy(prefix_cache) if prefix_cache is not None else make_prompt_cache(model)
    body = ids[n_prefix:]
    if body.shape[0] > 1:
        model(body[:-1][None], cache=cache)
        mx.eval([c.state for c in cache])
    out = model(body[-1:][None], cache=cache)[:, -1, :]
    mx.eval(out); return out

rng = mx.random.key(0)
print(f"model={MODEL}  (timings: best of 3, ms)")
print(f"{'N':>6} {'naive':>9} {'split':>9} {'ratio':>6} {'split tok/s':>12} {'prefix256':>10}  argmax-agree")
for N in (128, 256, 512, 1024, 2048):
    ids = mx.random.randint(1000, 200000, (N,), key=mx.random.key(N))
    a = naive(ids); b = split(ids)
    agree = int(mx.argmax(a).item() == mx.argmax(b).item())
    maxdiff = float(mx.abs(a - b).max().item())
    tn = t(lambda: naive(ids)); ts = t(lambda: split(ids))
    # prefix-cached variant: first 256 tokens cached once
    P = min(256, N - 2)
    pc = make_prompt_cache(model); model(ids[:P][None], cache=pc); mx.eval([c.state for c in pc])
    tp = t(lambda: split(ids, pc, P))
    print(f"{N:>6} {tn:>9.1f} {ts:>9.1f} {tn/ts:>6.2f} {N/ts*1e3:>12.0f} {tp:>10.1f}  {agree} (max|dlogit|={maxdiff:.3g})")
