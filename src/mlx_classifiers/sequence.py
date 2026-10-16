"""Sequence-scoring readout (``options.readout = "sequence"``, ADR-0003).

Each option's full text is teacher-forced after the prompt and scored by its
length-normalised loglikelihood (mean log-probability per token). The prompt
minus its last token is prefilled once and cached; each option then costs one
small pass over ``[last prompt token] + option tokens``, reading logits at
exactly the positions that predict the option's tokens. Options may share
first tokens or be many tokens long.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .calibrate import token_logprob
from .errors import LabelTokenizationError
from .prompt import CompiledPrompt
from .readout import label_token_ids
from .runner import Runner

__all__ = ["SequenceScores", "score_sequences"]


@dataclass(frozen=True)
class SequenceScores:
    """Per-option results in option order. ``scores`` are the length-normalised
    loglikelihoods used as the answer's logits."""

    options: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    loglikelihoods: tuple[float, ...]
    scores: tuple[float, ...]
    readout_position: int
    input_tokens: int
    cached_tokens: int

    @property
    def option_token_ids(self) -> dict[str, list[int]]:
        return {o: list(ids) for o, ids in zip(self.options, self.token_ids, strict=True)}


def score_sequences(
    prompt: CompiledPrompt,
    runner: Runner,
    tokenize: Callable[[str], list[int]] | None = None,
) -> SequenceScores:
    tokenize = tokenize or runner.tokenize
    tokens = prompt.tokens
    n = len(tokens)
    conts = []
    for i in range(len(prompt.options)):
        ids = label_token_ids(prompt.text, prompt.label_for(i), tokenize, tokens)
        conts.append(ids)
    if len(set(conts)) != len(conts):
        raise LabelTokenizationError(
            f"options {list(prompt.options)} include identically tokenized texts"
        )
    kw = {"images": prompt.images} if prompt.images else {}
    cache = runner.cache_prefix(tokens[:-1], **kw)
    lls, scores = [], []
    input_tokens, cached_tokens = 0, 0
    for ids in conts:
        positions = range(n - 1, n - 1 + len(ids))  # position p predicts token p + 1
        readout = runner.prefill(tokens + ids, cache, positions, **kw)
        ll = math.fsum(token_logprob(readout.at(p), t) for p, t in zip(positions, ids, strict=True))
        lls.append(ll)
        scores.append(ll / len(ids))
        input_tokens += readout.input_tokens
        cached_tokens += readout.cached_tokens
    return SequenceScores(
        options=prompt.options,
        token_ids=tuple(conts),
        loglikelihoods=tuple(lls),
        scores=tuple(scores),
        readout_position=prompt.readout_position,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
    )
