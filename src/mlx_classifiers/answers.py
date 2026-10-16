"""Build typed answers from a readout row and an option binding."""

from __future__ import annotations

from .calibrate import gather, softmax
from .prompt import CompiledPrompt
from .readout import OptionBinding
from .schema import ChoiceAnswer, ChoiceQuestion, Debug, NoulAnswer, NoulQuestion, ScoreAnswer

__all__ = ["build_answer"]


def _argmax(xs: list[float]) -> int:
    return max(range(len(xs)), key=lambda i: (xs[i], -i))  # first index wins ties


def build_answer(
    prompt: CompiledPrompt,
    binding: OptionBinding,
    row,
    *,
    temperature: float = 1.0,
    debug: bool = False,
) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    """Gather the option logits from ``row``, softmax over them only, apply
    ``temperature``, and shape the answer for the question type.

    ``raw_*`` are always the ``T = 1`` softmax; ``confidence`` is the calibrated
    probability of the returned option.
    """
    raw_logits = gather(row, binding.token_ids)
    raw = softmax(raw_logits)
    cal = softmax(raw_logits, temperature)
    dbg = (
        Debug(
            readout_position=binding.readout_position,
            option_token_ids=binding.option_token_ids,
            raw_logits=dict(zip(binding.options, raw_logits, strict=True)),
            temperature=temperature,
        )
        if debug
        else None
    )
    question = prompt.question
    if isinstance(question, NoulQuestion):
        return NoulAnswer(type="noul", noul=cal[0], raw_noul=raw[0], debug=dbg)
    best = _argmax(cal)
    probabilities = dict(zip(binding.options, cal, strict=True))
    raw_probabilities = dict(zip(binding.options, raw, strict=True))
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            type="choice",
            choice=binding.options[best],
            probabilities=probabilities,
            raw_probabilities=raw_probabilities,
            confidence=cal[best],
            debug=dbg,
        )
    return ScoreAnswer(
        type="score",
        score=best,
        expected_score=sum(i * p for i, p in enumerate(cal)),
        legend={str(i): level for i, level in enumerate(question.criteria)},
        probabilities=probabilities,
        raw_probabilities=raw_probabilities,
        confidence=cal[best],
        debug=dbg,
    )
