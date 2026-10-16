"""Story 3.6: where does one delta-cached decision's wall time go?

Usage: .venv/bin/python proto/delta_floor.py [MODEL] [STATE_TOKENS] [TAIL] [stock]
(``stock``: MLXRunner.load(moe_kernel=False), the pre-3.6 expert path)

Builds the bench's delta-tail prompts (state of STATE_TOKENS tokens, last TAIL
changing), warms the runner's delta store with real decisions, then replays
``MLXRunner.prefill`` phase by phase with ``mx.synchronize()`` fences and
reports the median of each phase over the timed decisions. The untimed
decision (``runner.prefill`` + ``build_answer``) is timed alongside for the
total, so the phase sum can be checked against it.
"""
import statistics
import sys
import time

import mlx.core as mx

from mlx_classifiers.answers import build_answer
from mlx_classifiers.bench import PREFIX, QUESTION, delta_states
from mlx_classifiers.mlx_runner import MLXRunner, _mx_row, cache_nbytes, rewind_cache
from mlx_classifiers.prompt import compile_question
from mlx_classifiers.readout import bind_options
from mlx_classifiers.schema import SystemOneRequest

model_id = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
n_state = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
tail = int(sys.argv[3]) if len(sys.argv) > 3 else 50
WARM, REPS = 3, 7

stock = len(sys.argv) > 4 and sys.argv[4] == "stock"
runner = MLXRunner.load(model_id, moe_kernel=not stock)
states = delta_states(runner.tokenize, n_state, tail, WARM + 2 * REPS)
prompts = []
for st in states:
    req = SystemOneRequest.model_validate({"prefix": PREFIX, "state": st, "questions": {"q": QUESTION}})
    p = compile_question(req, "q", template=runner.chat_template, tokenize=runner.tokenize)
    prompts.append((p, bind_options(p, runner.tokenize)))
pc = runner.cache_prefix(prompts[0][0].prefix_tokens, delta=False)
runner.prompt_cache.clear()


def whole(p, b):
    mx.synchronize()
    t = time.perf_counter()
    r = runner.prefill(p.tokens, pc, [p.readout_position])
    build_answer(p, b, r.at(p.readout_position))
    return (time.perf_counter() - t) * 1e3


def phased(p, b):
    """MLXRunner.prefill inlined, one fence per phase."""
    ph = {}
    store = runner.prompt_cache

    def mark(name, t0):
        mx.synchronize()
        ph[name] = (time.perf_counter() - t0) * 1e3
        return time.perf_counter()

    tokens = tuple(p.tokens)
    p0 = p1 = p.readout_position
    t = time.perf_counter()
    # lookup: candidate scan + rewind (deepcopy + trim), forced to evaluate
    hit, cache = store.lookup(tokens, rewind_cache, max_len=p0, min_len=len(pc))
    mx.eval([c.state for c in cache])
    t = mark("lookup (scan + deepcopy + trim)", t)
    runner._cache_only(tokens[hit:p0], cache)
    t = mark("cache-only prefill (delta)", t)
    import copy

    snap = copy.deepcopy(cache)
    mx.eval([c.state for c in snap])
    t = mark("store: deepcopy", t)
    nb = cache_nbytes(snap)
    t = mark("store: cache_nbytes", t)
    store.insert(tokens[:p0], snap, nb)
    t = mark("store: insert (+evict)", t)
    hidden = runner._backbone(mx.array(tokens[p0 : p1 + 1])[None], cache=cache)
    mx.eval(hidden)
    t = mark("readout: L=1 backbone", t)
    logits = runner._head(hidden[:, mx.array([0]), :])[0].astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)
    mx.eval(logits, lse)
    t = mark("readout: head + logsumexp", t)
    row = _mx_row(logits[0], lse[0], logits.shape[-1])
    from mlx_classifiers.runner import Readout

    r = Readout(positions=(p0,), logits=(row,), input_tokens=p1 + 1 - hit, cached_tokens=hit)
    build_answer(p, b, r.at(p0))
    mark("answer (masked softmax)", t)
    return ph, hit, nb


for p, b in prompts[:WARM]:
    whole(p, b)
totals, phases, info = [], [], None
rest = prompts[WARM:]
for i in range(REPS):
    totals.append(whole(*rest[2 * i]))
    ph, hit, nb = phased(*rest[2 * i + 1])
    phases.append(ph)
    info = (hit, nb)

print(f"moe kernel layers {runner.moe_kernel_layers}")
print(f"model {model_id}  state {n_state}  tail {tail}  hit {info[0]}  entry {info[1] / 2**20:.0f} MiB")
print(f"store: {len(runner.prompt_cache)} entries, {runner.prompt_cache.nbytes / 2**20:.0f} MiB")
total = statistics.median(totals)
s = 0.0
print(f"| phase | p50 ms | share |\n|---|---|---|")
for k in phases[0]:
    v = statistics.median(ph[k] for ph in phases)
    s += v
    print(f"| {k} | {v:.1f} | {100 * v / total:.0f}% |")
print(f"| sum of phases | {s:.1f} | {100 * s / total:.0f}% |")
print(f"| whole decision (runner.prefill + answer) | {total:.1f} | 100% |")
