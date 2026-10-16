"""``decide``: request in, typed answers out, through any :class:`Runner`.

Epic 2/3 supply a real Runner; nothing here knows about weights.
"""

from __future__ import annotations

import time

from .answers import build_answer
from .prompt import compile_request
from .readout import bind_options
from .runner import Runner
from .schema import SystemOneRequest, SystemOneResponse, Timing, Usage

__all__ = ["decide"]


def decide(
    request: SystemOneRequest, runner: Runner, *, temperature: float | None = None
) -> SystemOneResponse:
    """Answer every question in ``request`` with one prefill each and no decode.

    ``temperature`` is the fitted calibration temperature for this runner's model
    and readout mode; it is applied when ``options.calibrated`` is true (the
    default). Without one, calibrated equals raw.

    The ``prefix`` is KV-cached once per call and shared by every question.
    ``usage.input_tokens`` sums the tokens each prefill computed (suffixes only
    when cached) and ``cached_tokens`` the prefix tokens each one reused; the
    one-off ``cache_prefix`` cost is not counted.
    """
    t0 = time.perf_counter()
    opts = request.effective_options
    if opts.readout != "first-token":
        raise NotImplementedError("sequence readout lands in story 1.6")
    if opts.permutations != 1:
        raise NotImplementedError("option-order permutation lands in story 1.7")
    applied = temperature if (opts.calibrated and temperature is not None) else 1.0

    prompts = compile_request(request, runner)
    bindings = [bind_options(p, runner.tokenize) for p in prompts]
    cache = None
    if prompts[0].prefix_len:
        cache = runner.cache_prefix(prompts[0].prefix_tokens)

    answers, input_tokens, cached_tokens = {}, 0, 0
    prefill_s = readout_s = 0.0
    for prompt, binding in zip(prompts, bindings, strict=True):
        t1 = time.perf_counter()
        readout = runner.prefill(prompt.tokens, cache, [prompt.readout_position])
        t2 = time.perf_counter()
        answers[prompt.name] = build_answer(
            prompt,
            binding,
            readout.at(prompt.readout_position),
            temperature=applied,
            debug=opts.debug,
        )
        readout_s += time.perf_counter() - t2
        prefill_s += t2 - t1
        input_tokens += readout.input_tokens
        cached_tokens += readout.cached_tokens

    return SystemOneResponse(
        model=request.model or getattr(runner, "model_id", "default"),
        answers=answers,
        usage=Usage(input_tokens=input_tokens, output_tokens=0, cached_tokens=cached_tokens),
        timing_ms=Timing(
            prefill=prefill_s * 1e3,
            readout=readout_s * 1e3,
            total=(time.perf_counter() - t0) * 1e3,
        ),
    )
