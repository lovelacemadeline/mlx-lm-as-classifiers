"""Compile a request into one token sequence per question, with a known readout position.

Layout (separate mode: one prompt per question; packed mode is out of scope):

- system turn = ``request.prefix`` -- the cacheable head, byte-identical across
  requests whatever the ``state``;
- user turn = the rendered ``state`` plus one question with its labelled options;
- the assistant turn is opened, so the next token *is* the answer label. The
  readout position is the last token of the sequence (the end of the template's
  assistant-start marker): its logits are the distribution over that label.
"""

from __future__ import annotations

import json
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import PromptError, TooManyOptionsError
from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from .template import ChatTemplate

__all__ = [
    "LETTER_LABELS",
    "NOUL_LABELS",
    "NOUL_OPTIONS",
    "CompiledPrompt",
    "compile_question",
    "compile_request",
    "option_labels",
    "question_options",
    "render_state",
    "render_user_turn",
]

LETTER_LABELS = tuple(string.ascii_uppercase)
NOUL_OPTIONS = ("yes", "no")  # internal option names; the answer reports P(yes)
NOUL_LABELS = ("Yes", "No")  # the fixed label pair the model is asked to emit

QuestionT = NoulQuestion | ChoiceQuestion | ScoreQuestion
Tokenize = Callable[[str], list[int]]


def question_options(question: QuestionT) -> tuple[str, ...]:
    """Option names in declared order: criteria keys for ``choice``, the level
    indices (``"0"``, ``"1"``, ...) for ``score``, ``("yes", "no")`` for ``noul``."""
    if isinstance(question, NoulQuestion):
        return NOUL_OPTIONS
    if isinstance(question, ChoiceQuestion):
        return tuple(question.criteria)
    return tuple(str(i) for i in range(len(question.criteria)))


def option_labels(question: QuestionT) -> tuple[str, ...]:
    """The label the model emits for each option: Yes/No, else letters A..Z."""
    if isinstance(question, NoulQuestion):
        return NOUL_LABELS
    n = len(question_options(question))
    if n > len(LETTER_LABELS):
        raise TooManyOptionsError(
            f"{n} options exceeds the {len(LETTER_LABELS)} single-letter labels; "
            "split the question or use independent per-option scoring (a future story)"
        )
    return LETTER_LABELS[:n]


def render_state(state: str | dict[str, Any] | list[str]) -> str:
    """A string as-is; an object as sorted-key pretty JSON; a list as numbered blocks."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)
    return "\n\n".join(f"[{i}] {block}" for i, block in enumerate(state, start=1))


def render_user_turn(
    state: str | dict[str, Any] | list[str],
    question: QuestionT,
    labels: Sequence[str] | None = None,
    order: Sequence[int] | None = None,
) -> str:
    """The user turn for one question. ``order`` lists option indices in display
    order (default: declared order); ``labels[k]`` labels the k-th displayed option."""
    labels = tuple(labels) if labels is not None else option_labels(question)
    lines = ["State:", render_state(state), "", f"Question: {question.instructions}", ""]
    if isinstance(question, NoulQuestion):
        lines.append(f"Answer {labels[0]} or {labels[1]} only.")
        return "\n".join(lines)
    if isinstance(question, ChoiceQuestion):
        items = list(question.criteria.items())
        entries = [f"{name}: {desc}" if desc else name for name, desc in items]
        lines.append("Options:")
        noun = "option"
    else:
        entries = list(question.criteria)
        lines.append("Scale (lowest to highest):")
        noun = "level"
    order = tuple(order) if order is not None else tuple(range(len(entries)))
    if sorted(order) != list(range(len(entries))):
        raise PromptError(f"order {order} is not a permutation of {len(entries)} options")
    lines.extend(f"{labels[k]}. {entries[i]}" for k, i in enumerate(order))
    lines += ["", f"Answer with the letter of one {noun} only."]
    return "\n".join(lines)


@dataclass(frozen=True)
class CompiledPrompt:
    """One question's prompt. ``tokens[:prefix_len]`` is the cacheable system turn;
    ``readout_position`` indexes the token whose logits pick the label.

    ``labels[k]`` is the label shown for ``options[order[k]]``."""

    name: str
    question: QuestionT
    text: str
    tokens: tuple[int, ...]
    prefix_len: int
    readout_position: int
    options: tuple[str, ...]
    labels: tuple[str, ...]
    order: tuple[int, ...]

    @property
    def prefix_tokens(self) -> tuple[int, ...]:
        return self.tokens[: self.prefix_len]

    def label_for(self, option_index: int) -> str:
        return self.labels[self.order.index(option_index)]


def compile_question(
    request: SystemOneRequest,
    name: str,
    *,
    template: ChatTemplate,
    tokenize: Tokenize,
    order: Sequence[int] | None = None,
) -> CompiledPrompt:
    """Render and tokenize one question of ``request``."""
    question = request.questions[name]
    options = question_options(question)
    labels = option_labels(question)
    if isinstance(question, NoulQuestion):
        order = (0, 1)  # the Yes/No pair is fixed, never permuted
    order = tuple(order) if order is not None else tuple(range(len(options)))
    user = render_user_turn(request.state, question, labels, order)
    chat = template.render(request.prefix, user)
    tokens = tuple(tokenize(chat.text))
    prefix_len = 0
    if chat.prefix_text:
        prefix_ids = tuple(tokenize(chat.prefix_text))
        if tokens[: len(prefix_ids)] != prefix_ids:
            raise PromptError(
                "the prefix does not tokenize to a token-aligned head of the prompt; "
                "it cannot be KV-cached"
            )
        prefix_len = len(prefix_ids)
    if len(tokens) <= prefix_len:
        raise PromptError("prompt has no tokens after the prefix")
    return CompiledPrompt(
        name=name,
        question=question,
        text=chat.text,
        tokens=tokens,
        prefix_len=prefix_len,
        readout_position=len(tokens) - 1,
        options=options,
        labels=labels,
        order=order,
    )


def compile_request(request: SystemOneRequest, runner: Any) -> list[CompiledPrompt]:
    """One :class:`CompiledPrompt` per question, in request order, via the runner's
    ``chat_template`` and ``tokenize``."""
    return [
        compile_question(request, name, template=runner.chat_template, tokenize=runner.tokenize)
        for name in request.questions
    ]
