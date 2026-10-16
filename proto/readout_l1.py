"""Story 2.9: the L=1 readout pass, timed alone.

Usage: .venv/bin/python proto/readout_l1.py MODEL [STATE_TOKENS] [REPS] [int8 [pad]]
(``pad``: route the INT8 wrappers' short calls to the padded 64-row GEMM instead of
the GEMV, ``wiring.GEMV_MAX_ROWS = 0``; the other story 2.9 candidate)

Builds the bench prompt at STATE_TOKENS state tokens, prefills its cache-only rows
once, then times the L=1 backbone pass over the readout row (the pass the INT8
wrappers' under-64-row path serves) and, separately, the head: a fresh copy of the
cache per rep (copied outside the timer), ``mx.synchronize()`` fences, median and
p10/p90 of REPS after 5 warm passes. Prints the loaded INT8 state and power.
"""

import copy
import statistics
import subprocess
import sys
import time

import mlx.core as mx

from mlx_classifiers.bench import PREFIX, QUESTION, filler_state
from mlx_classifiers.mlx_runner import MLXRunner
from mlx_classifiers.prompt import compile_question
from mlx_classifiers.schema import SystemOneRequest

model_id = sys.argv[1]
n_state = int(sys.argv[2]) if len(sys.argv) > 2 else 512
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 50
int8 = len(sys.argv) > 4 and sys.argv[4] == "int8"
pad = len(sys.argv) > 5 and sys.argv[5] == "pad"
if pad:
    from mlx_classifiers.int8 import wiring

    wiring.GEMV_MAX_ROWS = 0

runner = MLXRunner.load(model_id, int8=int8, prompt_cache=None)
state, _ = filler_state(runner.tokenize, n_state)
req = SystemOneRequest.model_validate(
    {"prefix": PREFIX, "state": state, "questions": {"q": QUESTION}}
)
p = compile_question(req, "q", template=runner.chat_template, tokenize=runner.tokenize)
p0 = p.readout_position
from mlx_lm.models.cache import make_prompt_cache

base = make_prompt_cache(runner.model)
runner._cache_only(p.tokens[:p0], base)
row = mx.array(p.tokens[p0 : p0 + 1])[None]


def once():
    c = copy.deepcopy(base)
    mx.eval([x.state for x in c])
    mx.synchronize()
    t = time.perf_counter()
    h = runner._backbone(row, cache=c)
    mx.eval(h)
    mx.synchronize()
    t1 = time.perf_counter()
    logits = runner._head(h[:, -1:, :]).astype(mx.float32)
    mx.eval(mx.logsumexp(logits, axis=-1))
    mx.synchronize()
    return (t1 - t) * 1e3, (time.perf_counter() - t1) * 1e3


for _ in range(5):
    once()
bb, hd = zip(*(once() for _ in range(reps)), strict=True)
q = statistics.quantiles(bb, n=10)
power = subprocess.run(
    ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
).stdout.split("\n")[0]
lpm = subprocess.run(["pmset", "-g"], capture_output=True, text=True, check=False).stdout
lpm = next((ln.split()[-1] for ln in lpm.splitlines() if "lowpowermode" in ln), "?")
small = "pad64" if pad else "gemv"
print(f"model {model_id} int8 {runner.int8} ({small}) expert_layers {runner.int8_expert_layers}")
print(f"prompt {len(p.tokens)} tok, readout at {p0}; {reps} reps; lowpowermode {lpm}; {power}")
print(f"L=1 backbone ms: p50 {statistics.median(bb):.2f}  p10 {q[0]:.2f}  p90 {q[-1]:.2f}")
print(f"head + logsumexp ms: p50 {statistics.median(hd):.2f}")
