"""Build typed answers from per-option scores."""

from __future__ import annotations

from collections.abc import Sequence

from .calibrate import gather, softmax
from .prompt import CompiledPrompt, QuestionT
from .readout import OptionBinding
from .schema import ChoiceAnswer, ChoiceQuestion, Debug, NoulAnswer, NoulQuestion, ScoreAnswer

__all__ = ["build_answer", "shape_answer"]


def _argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=lambda i: (xs[i], -i))  # first index wins ties


def build_answer(
    prompt: CompiledPrompt,
    binding: OptionBinding,
    row,
    *,
    temperature: float = 1.0,
    debug: bool = False,
) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    """First-token answer: gather the option logits from ``row``, softmax over them
    only, apply ``temperature``, and shape the answer for the question type."""
    raw_logits = gather(row, binding.token_ids)
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
    return shape_answer(
        prompt.question,
        binding.options,
        softmax(raw_logits),
        softmax(raw_logits, temperature),
        debug=dbg,
    )


def shape_answer(
    question: QuestionT,
    options: Sequence[str],
    raw: Sequence[float],
    calibrated: Sequence[float],
    *,
    debug: Debug | None = None,
) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    """Shape option distributions (in option order) into the typed answer.

    ``raw`` is the ``T = 1`` distribution, ``calibrated`` the temperature-scaled
    one; ``confidence`` is the calibrated probability of the returned option.
    """
    raw, cal = list(raw), list(calibrated)
    if isinstance(question, NoulQuestion):
        return NoulAnswer(type="noul", noul=cal[0], raw_noul=raw[0], debug=debug)
    best = _argmax(cal)
    probabilities = dict(zip(options, cal, strict=True))
    raw_probabilities = dict(zip(options, raw, strict=True))
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            type="choice",
            choice=options[best],
            probabilities=probabilities,
            raw_probabilities=raw_probabilities,
            confidence=cal[best],
            debug=debug,
        )
    return ScoreAnswer(
        type="score",
        score=best,
        expected_score=sum(i * p for i, p in enumerate(cal)),
        legend={str(i): level for i, level in enumerate(question.criteria)},
        probabilities=probabilities,
        raw_probabilities=raw_probabilities,
        confidence=cal[best],
        debug=debug,
    )
