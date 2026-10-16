"""Where does the per-decision floor go? Time the pieces of the split path at N=256."""
import time, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, tok = load("mlx-community/gemma-4-E2B-it-qat-4bit")
def t(fn, reps=5):
    fn(); mx.synchronize(); b=1e9
    for _ in range(reps):
        s=time.perf_counter(); fn(); mx.synchronize(); b=min(b,time.perf_counter()-s)
    return b*1e3
ids = mx.random.randint(1000, 200000, (256,), key=mx.random.key(1))
def chunk_only():
    c = make_prompt_cache(model); model(ids[:-1][None], cache=c); mx.eval([x.state for x in c])
c = make_prompt_cache(model); model(ids[:-1][None], cache=c); mx.eval([x.state for x in c])
import copy
def last_full():
    cc = copy.deepcopy(c); out = model(ids[-1:][None], cache=cc)[:, -1, :]; mx.eval(out)
def last_hidden_only():
    cc = copy.deepcopy(c); h = model.language_model.model(ids[-1:][None], cache=cc); mx.eval(h)
def lm_head_only():
    h = mx.zeros((1,1,1536), dtype=mx.bfloat16); out = model.language_model.model.embed_tokens.as_linear(h); mx.eval(out)
def deepcopy_only():
    cc = copy.deepcopy(c); mx.eval([x.state for x in cc])
print(f"prefill 255 tok cache-only : {t(chunk_only):7.1f} ms")
print(f"last token, all layers+head: {t(last_full):7.1f} ms")
print(f"  of which hidden (35 layers): {t(last_hidden_only):7.1f} ms")
print(f"  of which lm_head (262k x1536): {t(lm_head_only):7.1f} ms")
print(f"  deepcopy cache overhead     : {t(deepcopy_only):7.1f} ms")
