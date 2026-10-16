"""Option binding and the prefix-tree first-token readout (ADR-0003).

Each option is bound to a label (``prompt.option_labels``) and the label is
resolved to token ids *in context*: the ids it gets when emitted right after
the compiled prompt, which picks the right leading-space variant for the
template position. A prefix tree over those id sequences reports the first
depth at which every label is distinct. First-token mode needs depth one *and*
one token per label; anything else is a :class:`LabelTokenizationError`, never
a silent fallback. Use ``options.readout = "sequence"`` for such options.

With ``options.label_style = "option-text"`` (story 1.8) the labels are the
option names themselves, so the same rules apply to the names; the error then
also points back at letter labels.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .errors import LabelTokenizationError
from .prompt import CompiledPrompt

__all__ = [
    "OptionBinding",
    "PrefixTree",
    "bind_labels",
    "bind_options",
    "emitted_label",
    "label_token_ids",
]

Tokenize = Callable[[str], list[int]]
SEQUENCE_HINT = 'use options.readout = "sequence", or labels that are single tokens'
OPTION_TEXT_HINT = (
    'use options.readout = "sequence", or options.label_style = "letters" '
    "(or option names that are single tokens)"
)


def emitted_label(context_text: str, label: str) -> str:
    """The text the model would emit for ``label`` after ``context_text``: the bare
    label after whitespace (e.g. a template's trailing newline), else ``" " + label``."""
    return label if not context_text or context_text[-1].isspace() else " " + label


def label_token_ids(
    context_text: str,
    label: str,
    tokenize: Tokenize,
    context_tokens: Sequence[int] | None = None,
    hint: str = SEQUENCE_HINT,
) -> tuple[int, ...]:
    """Token ids of ``label`` as it would appear right after ``context_text``.

    Raises :class:`LabelTokenizationError` if appending the label re-tokenizes the
    context (the label merges with the readout token), since then no next-token
    distribution at the readout position can represent it.
    """
    base = tuple(context_tokens) if context_tokens is not None else tuple(tokenize(context_text))
    full = tuple(tokenize(context_text + emitted_label(context_text, label)))
    if full[: len(base)] != base or len(full) == len(base):
        raise LabelTokenizationError(
            f"label {label!r} merges with the token at the readout position; {hint}"
        )
    return full[len(base) :]


@dataclass(frozen=True)
class PrefixTree:
    """A prefix tree over label token sequences, summarised.

    ``depth`` is the first token depth at which every label's sequence is distinct
    (1 = the first token discriminates; a label that has ended counts as distinct
    from one that continues), or ``None`` if two labels tokenize identically. ``collisions`` groups labels that share a
    first token.
    """

    sequences: Mapping[str, tuple[int, ...]]
    depth: int | None
    collisions: tuple[tuple[str, ...], ...]

    @classmethod
    def build(cls, sequences: Mapping[str, Sequence[int]]) -> PrefixTree:
        seqs = {label: tuple(ids) for label, ids in sequences.items()}
        first: dict[int, list[str]] = defaultdict(list)
        for label, ids in seqs.items():
            first[ids[0]].append(label)
        collisions = tuple(tuple(g) for g in first.values() if len(g) > 1)
        depth = None
        for d in range(1, max(map(len, seqs.values())) + 1):
            heads = [ids[:d] for ids in seqs.values()]  # a shorter label ends early
            if len(set(heads)) == len(heads):
                depth = d
                break
        return cls(sequences=seqs, depth=depth, collisions=collisions)


@dataclass(frozen=True)
class OptionBinding:
    """Options bound to single-token labels at one readout position.

    ``token_ids[i]`` is the mask id for ``options[i]``; ``labels[i]`` its label.
    """

    options: tuple[str, ...]
    labels: tuple[str, ...]
    token_ids: tuple[int, ...]
    readout_position: int
    tree: PrefixTree

    @property
    def option_token_ids(self) -> dict[str, list[int]]:
        """``{option: [token id]}``, the shape of ``debug.option_token_ids``."""
        return {o: [t] for o, t in zip(self.options, self.token_ids, strict=True)}


def bind_labels(
    labels: Sequence[str],
    context_text: str,
    tokenize: Tokenize,
    context_tokens: Sequence[int] | None = None,
    hint: str = SEQUENCE_HINT,
) -> tuple[tuple[int, ...], PrefixTree]:
    """Resolve labels in context and require first-token readout to be possible.

    Returns one mask id per label plus the prefix tree. Raises
    :class:`LabelTokenizationError` naming the offending labels if two share a first
    token (distinguishing depth > 1) or a label is more than one token; ``hint``
    ends the message with what to do instead.
    """
    if len(set(labels)) != len(labels):
        raise LabelTokenizationError(f"duplicate labels {list(labels)}")
    base = tuple(context_tokens) if context_tokens is not None else tuple(tokenize(context_text))
    seqs = {lab: label_token_ids(context_text, lab, tokenize, base, hint) for lab in labels}
    tree = PrefixTree.build(seqs)
    if tree.collisions:
        names = "; ".join(" and ".join(repr(x) for x in g) for g in tree.collisions)
        where = f"depth {tree.depth}" if tree.depth else "no depth (identical token ids)"
        raise LabelTokenizationError(
            f"labels {names} share a first token; they first differ at {where}, "
            f"which first-token mode cannot read; {hint}"
        )
    multi = [lab for lab, ids in seqs.items() if len(ids) != 1]
    if multi:
        detail = ", ".join(f"{lab!r} -> {len(seqs[lab])} tokens {list(seqs[lab])}" for lab in multi)
        raise LabelTokenizationError(f"labels are not single tokens: {detail}; {hint}")
    return tuple(seqs[lab][0] for lab in labels), tree


def bind_options(prompt: CompiledPrompt, tokenize: Tokenize) -> OptionBinding:
    """Bind ``prompt``'s options to mask ids at its readout position. With
    ``label_style = "option-text"`` the labels are the option names (Yes/No for
    ``noul``), and an error points at sequence mode or letter labels."""
    labels = tuple(prompt.label_for(i) for i in range(len(prompt.options)))
    hint = OPTION_TEXT_HINT if prompt.label_style == "option-text" else SEQUENCE_HINT
    ids, tree = bind_labels(labels, prompt.text, tokenize, prompt.tokens, hint)
    return OptionBinding(
        options=prompt.options,
        labels=labels,
        token_ids=ids,
        readout_position=prompt.readout_position,
        tree=tree,
    )
