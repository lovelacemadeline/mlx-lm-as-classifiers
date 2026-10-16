"""``decide``: request in, typed answers out, through any :class:`Runner`.

Epic 2/3 supply a real Runner; nothing here knows about weights.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .answers import shape_answer
from .calibrate import gather, softmax
from .prompt import CompiledPrompt, compile_question
from .readout import bind_options
from .runner import PrefixCache, Runner
from .schema import ChoiceQuestion, Debug, SystemOneRequest, SystemOneResponse, Timing, Usage
from .sequence import score_sequences

__all__ = ["decide", "permutation_orders"]


def permutation_orders(n: int, count: int, seed: str = "") -> list[tuple[int, ...]]:
    """``count`` display orders over ``n`` options, deterministically.

    Order 0 is the declared order. Orders ``1 .. n-1`` are the cyclic rotations
    (so with ``count <= n`` every option visits a different position each time);
    beyond that, shuffles seeded by ``(seed, k)``.
    """
    orders: list[tuple[int, ...]] = []
    for k in range(count):
        if k < n:
            orders.append(tuple((i + k) % n for i in range(n)))
        else:
            perm = list(range(n))
            random.Random(f"{seed}:{k}").shuffle(perm)
            orders.append(tuple(perm))
    return orders


@dataclass
class _Scored:
    """One compiled ordering's per-option scores, in option (not display) order."""

    prompt: CompiledPrompt
    logits: list[float]
    token_ids: dict[str, list[int]]
    input_tokens: int
    cached_tokens: int


def _score(
    prompt: CompiledPrompt, runner: Runner, cache: PrefixCache | None, readout: str
) -> _Scored:
    if readout == "sequence":
        s = score_sequences(prompt, runner)
        return _Scored(prompt, list(s.scores), s.option_token_ids, s.input_tokens, s.cached_tokens)
    binding = bind_options(prompt, runner.tokenize)
    r = runner.prefill(prompt.tokens, cache, [prompt.readout_position])
    logits = gather(r.at(prompt.readout_position), binding.token_ids)
    return _Scored(prompt, logits, binding.option_token_ids, r.input_tokens, r.cached_tokens)


def decide(
    request: SystemOneRequest, runner: Runner, *, temperature: float | None = None
) -> SystemOneResponse:
    """Answer every question in ``request`` with prefill only, no decode.

    ``temperature`` is the fitted calibration temperature for this runner's model
    and readout mode; it is applied when ``options.calibrated`` is true (the
    default). Without one, calibrated equals raw.

    First-token mode: the ``prefix`` is KV-cached once per call and shared by
    every prompt; each prompt is one prefill with logits at its readout position.
    Sequence mode: each prompt caches itself and scores every option with one
    small pass (:mod:`mlx_classifiers.sequence`).

    ``options.permutations = N`` compiles each ``choice`` question N times with
    the option order permuted (:func:`permutation_orders`), labels re-bound per
    ordering, and returns the mean of the N distributions (raw and calibrated
    averaged separately). ``noul`` (fixed Yes/No) and ``score`` (an ordered scale)
    are compiled once. With ``debug``, ``permutation_spread`` gives each option's
    max minus min calibrated probability across orderings.

    ``usage.input_tokens`` sums the tokens each prefill computed (suffixes only
    when cached) and ``cached_tokens`` the cached tokens each one reused; one-off
    ``cache_prefix`` costs are not counted.
    """
    t0 = time.perf_counter()
    opts = request.effective_options
    readout = opts.readout
    applied = temperature if (opts.calibrated and temperature is not None) else 1.0

    def compile_(name: str, order=None) -> CompiledPrompt:
        return compile_question(
            request,
            name,
            template=runner.chat_template,
            tokenize=runner.tokenize,
            order=order,
            readout=readout,
        )

    plans: dict[str, list[CompiledPrompt]] = {}
    for name, question in request.questions.items():
        if isinstance(question, ChoiceQuestion) and opts.permutations > 1:
            orders = permutation_orders(len(question.criteria), opts.permutations, seed=name)
            plans[name] = [compile_(name, order) for order in orders]
        else:
            plans[name] = [compile_(name)]

    first = next(iter(plans.values()))[0]
    cache = None
    if readout == "first-token" and first.prefix_len:
        cache = runner.cache_prefix(first.prefix_tokens)

    answers, input_tokens, cached_tokens = {}, 0, 0
    prefill_s = 0.0
    for name, prompts in plans.items():
        t1 = time.perf_counter()
        scored = [_score(p, runner, cache, readout) for p in prompts]
        prefill_s += time.perf_counter() - t1
        input_tokens += sum(s.input_tokens for s in scored)
        cached_tokens += sum(s.cached_tokens for s in scored)

        k = len(scored)
        options = scored[0].prompt.options
        raws = [softmax(s.logits) for s in scored]
        cals = [softmax(s.logits, applied) for s in scored]
        raw = [math.fsum(r[i] for r in raws) / k for i in range(len(options))]
        cal = [math.fsum(c[i] for c in cals) / k for i in range(len(options))]

        dbg = None
        if opts.debug:
            if readout == "first-token":  # one label id per ordering
                ids = {o: [i for s in scored for i in s.token_ids[o]] for o in options}
            else:  # option texts tokenize identically in every ordering
                ids = scored[0].token_ids
            dbg = Debug(
                readout_position=scored[0].prompt.readout_position,
                option_token_ids=ids,
                raw_logits={
                    o: math.fsum(s.logits[i] for s in scored) / k for i, o in enumerate(options)
                },
                temperature=applied,
                permutation_spread=(
                    {
                        o: max(c[i] for c in cals) - min(c[i] for c in cals)
                        for i, o in enumerate(options)
                    }
                    if k > 1
                    else None
                ),
            )
        answers[name] = shape_answer(request.questions[name], options, raw, cal, debug=dbg)

    total = (time.perf_counter() - t0) * 1e3
    return SystemOneResponse(
        model=request.model or getattr(runner, "model_id", "default"),
        answers=answers,
        usage=Usage(input_tokens=input_tokens, output_tokens=0, cached_tokens=cached_tokens),
        timing_ms=Timing(prefill=prefill_s * 1e3, readout=total - prefill_s * 1e3, total=total),
    )
