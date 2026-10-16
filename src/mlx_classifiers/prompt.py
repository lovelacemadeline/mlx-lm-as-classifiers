"""Compile a request into one token sequence per question, with a known readout position.

Layout (separate mode: one prompt per question; packed mode is out of scope):

- system turn = ``request.prefix`` -- the cacheable head, byte-identical across
  requests whatever the ``state``;
- user turn = the rendered ``state`` plus one question with its labelled options
  (letters ``A.``, ``B.``, ... by default; with ``options.label_style =
  "option-text"`` or in sequence mode, each option is shown by its own name);
- the assistant turn is opened, so the next token *is* the answer label. The
  readout position is the last token of the sequence (the end of the template's
  assistant-start marker): its logits are the distribution over that label.

Image blocks (story 6.1): a list state may hold ``{"type": "image/png",
"base64": ...}`` blocks. Each renders as ``[i] `` followed by the model's image
placeholder run (``boi + image * N + eoi``, :class:`~mlx_classifiers.images.ImageTokens`
from the runner's ``image_tokens``) at the block's position; the compiled prompt
records each run as an :class:`~mlx_classifiers.images.ImageSlot` (position,
length, PNG sha256). A state without image blocks compiles exactly as before.
"""

from __future__ import annotations

import json
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .errors import PromptError, TooManyOptionsError
from .images import ImageSlot, ImageTokens, decode_png_base64, find_image_runs
from .schema import ChoiceQuestion, ImageBlock, NoulQuestion, ScoreQuestion, SystemOneRequest
from .template import ChatTemplate

__all__ = [
    "LETTER_LABELS",
    "NOUL_LABELS",
    "NOUL_OPTIONS",
    "CompiledPrompt",
    "compile_question",
    "compile_request",
    "names_options",
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
State = str | dict[str, Any] | list[str | ImageBlock]

#: Stand-in for image ``k`` while the chat template renders (private-use code points,
#: which no template trims or escapes); replaced by the placeholder run afterwards.
_IMAGE_SENTINEL = "\ue000{}\ue001"


def question_options(question: QuestionT) -> tuple[str, ...]:
    """Option names in declared order: criteria keys for ``choice``, the level
    indices (``"0"``, ``"1"``, ...) for ``score``, ``("yes", "no")`` for ``noul``."""
    if isinstance(question, NoulQuestion):
        return NOUL_OPTIONS
    if isinstance(question, ChoiceQuestion):
        return tuple(question.criteria)
    return tuple(str(i) for i in range(len(question.criteria)))


def names_options(readout: str = "first-token", label_style: str = "letters") -> bool:
    """True when options are shown and answered by their own text rather than by
    letter: always in sequence mode, and in first-token mode with
    ``label_style = "option-text"`` (story 1.8)."""
    return readout == "sequence" or label_style == "option-text"


def option_labels(
    question: QuestionT, readout: str = "first-token", label_style: str = "letters"
) -> tuple[str, ...]:
    """The label the model emits for each option: Yes/No for ``noul``; otherwise
    letters A..Z, or the option text itself (criteria key or level) in sequence
    mode or with ``label_style = "option-text"``."""
    if isinstance(question, NoulQuestion):
        return NOUL_LABELS
    if names_options(readout, label_style):
        return tuple(question.criteria)  # choice: the names; score: the levels
    n = len(question_options(question))
    if n > len(LETTER_LABELS):
        raise TooManyOptionsError(
            f"{n} options exceeds the {len(LETTER_LABELS)} single-letter labels; "
            "split the question or use independent per-option scoring (a future story)"
        )
    return LETTER_LABELS[:n]


def render_state(state: State, image: Callable[[int, ImageBlock], str] | None = None) -> str:
    """A string as-is; an object as sorted-key pretty JSON; a list as numbered blocks.

    An image block renders as ``image(k, block)`` (``k`` counts images from 0);
    by default as ``<image k>``. Prompt compilation passes the placeholder."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)
    image = image or (lambda k, block: f"<image {k}>")
    parts, k = [], 0
    for i, block in enumerate(state, start=1):
        if isinstance(block, ImageBlock):
            parts.append(f"[{i}] {image(k, block)}")
            k += 1
        else:
            parts.append(f"[{i}] {block}")
    return "\n\n".join(parts)


def render_user_turn(
    state: State,
    question: QuestionT,
    labels: Sequence[str] | None = None,
    order: Sequence[int] | None = None,
    readout: str = "first-token",
    label_style: str = "letters",
    image: Callable[[int, ImageBlock], str] | None = None,
) -> str:
    """The user turn for one question. ``order`` lists option indices in display
    order (default: declared order); ``labels[k]`` labels the k-th displayed option.
    In sequence mode, and with ``label_style = "option-text"``, options are
    bulleted and answered by their own text (the two render identically).
    ``image`` renders image blocks (see :func:`render_state`)."""
    if labels is None:
        labels = option_labels(question, readout, label_style)
    labels = tuple(labels)
    state_text = render_state(state, image)
    lines = ["State:", state_text, "", f"Question: {question.instructions}", ""]
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
    if names_options(readout, label_style):
        lines.extend(f"- {entries[i]}" for i in order)
        lines += ["", f"Answer with one {noun} exactly as written."]
        return "\n".join(lines)
    lines.extend(f"{labels[k]}. {entries[i]}" for k, i in enumerate(order))
    lines += ["", f"Answer with the letter of one {noun} only."]
    return "\n".join(lines)


@dataclass(frozen=True)
class CompiledPrompt:
    """One question's prompt. ``tokens[:prefix_len]`` is the cacheable system turn;
    ``readout_position`` indexes the token whose logits pick the label.

    ``labels[k]`` is the label shown for ``options[order[k]]``. ``label_style`` is
    ``"letters"`` or ``"option-text"`` (meaningful in first-token mode only).
    ``images`` holds one :class:`~mlx_classifiers.images.ImageSlot` per image
    block, in state order (empty for text-only states)."""

    name: str
    question: QuestionT
    text: str
    tokens: tuple[int, ...]
    prefix_len: int
    readout_position: int
    options: tuple[str, ...]
    labels: tuple[str, ...]
    order: tuple[int, ...]
    readout: str = "first-token"
    label_style: str = "letters"
    images: tuple[ImageSlot, ...] = ()

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
    readout: str | None = None,
    label_style: str | None = None,
    image_tokens: ImageTokens | None = None,
) -> CompiledPrompt:
    """Render and tokenize one question of ``request``. ``readout`` and
    ``label_style`` default to ``request.options``. A state with image blocks
    needs ``image_tokens`` (the runner's placeholder convention)."""
    question = request.questions[name]
    readout = readout or request.effective_options.readout
    label_style = label_style or request.effective_options.label_style
    options = question_options(question)
    labels = option_labels(question, readout, label_style)
    if isinstance(question, NoulQuestion):
        order = (0, 1)  # the Yes/No pair is fixed, never permuted
    order = tuple(order) if order is not None else tuple(range(len(options)))
    if not isinstance(question, NoulQuestion) and names_options(readout, label_style):
        labels = tuple(labels[i] for i in order)  # labels are the texts: they follow the order
    pngs: list[bytes] = []
    image = None
    if request.has_images:
        if image_tokens is None:
            raise PromptError("the state holds image blocks but this runner has no vision path")
        budget = request.effective_options.vision_soft_tokens
        if budget is not None and budget != image_tokens.max_soft_tokens:
            image_tokens = replace(image_tokens, max_soft_tokens=budget)  # story 6.7
        pngs = [decode_png_base64(b.base64) for b in request.state if isinstance(b, ImageBlock)]
        image = lambda k, block: _IMAGE_SENTINEL.format(k)
    user = render_user_turn(request.state, question, labels, order, readout, label_style, image)
    chat = template.render(request.prefix, user)
    text = chat.text
    if pngs:
        for k, png in enumerate(pngs):
            marker = _IMAGE_SENTINEL.format(k)
            if text.count(marker) != 1 or marker in chat.prefix_text:
                raise PromptError("the chat template did not keep the image position")
            text = text.replace(marker, image_tokens.placeholder(png))
    tokens = tuple(tokenize(text))
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
    images = _image_slots(tokens, pngs, image_tokens) if pngs else ()
    return CompiledPrompt(
        name=name,
        question=question,
        text=text,
        tokens=tokens,
        prefix_len=prefix_len,
        readout_position=len(tokens) - 1,
        options=options,
        labels=labels,
        order=order,
        readout=readout,
        label_style=label_style,
        images=images,
    )


def _image_slots(
    tokens: Sequence[int], pngs: Sequence[bytes], spec: ImageTokens
) -> tuple[ImageSlot, ...]:
    """One slot per image: the placeholder runs, which must match the images one to one."""
    runs = find_image_runs(tokens, spec.image_token_id)
    if len(runs) != len(pngs):
        raise PromptError(
            f"{len(runs)} image-token runs in the prompt for {len(pngs)} image blocks "
            "(does the text itself contain the image placeholder?)"
        )
    slots = []
    for (start, length), png in zip(runs, pngs, strict=True):
        if length != spec.soft_tokens(png):
            raise PromptError(
                f"image run at {start} has {length} tokens, expected {spec.soft_tokens(png)}"
            )
        slots.append(ImageSlot.from_png(start, length, png, spec.max_soft_tokens))
    return tuple(slots)


def compile_request(
    request: SystemOneRequest, runner: Any, readout: str | None = None
) -> list[CompiledPrompt]:
    """One :class:`CompiledPrompt` per question, in request order, via the runner's
    ``chat_template`` and ``tokenize``."""
    return [
        compile_question(
            request,
            name,
            template=runner.chat_template,
            tokenize=runner.tokenize,
            readout=readout,
            image_tokens=getattr(runner, "image_tokens", None),
        )
        for name in request.questions
    ]
