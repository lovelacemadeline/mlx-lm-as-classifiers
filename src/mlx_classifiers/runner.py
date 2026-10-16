"""The Runner seam (ADR-0002) and a deterministic fake runner for weightless tests.

A :class:`Runner` is the only model-dependent piece of the pipeline. Everything
downstream (prompt compilation, label binding, readout, calibration) talks to it
through this protocol, and the Swift + vision port mirrors it one-to-one:

``tokenize(text) -> list[int]``
    Text to token ids. Special-token markup already present in ``text`` (from a
    chat template) is parsed as special tokens; nothing (no BOS) is added.

``cache_prefix(tokens) -> PrefixCache``
    Prefill ``tokens`` once and keep the KV state, so later calls only prefill
    the suffix. Produces no logits.

``chat_template``
    A :class:`~mlx_classifiers.template.ChatTemplate` for the model's tokenizer
    (rendering is text-only; ``tokenize`` turns it into ids).

``prefill(tokens, prefix_cache=None, readout_positions=None) -> Readout``
    Prefill the *full* sequence ``tokens``; when ``prefix_cache`` is given,
    ``tokens`` must start with ``prefix_cache.tokens`` and only the suffix is
    computed. Logits are produced **only** at ``readout_positions`` (absolute
    indices into ``tokens``, default: the last one), never for the whole
    sequence: the hidden state is sliced before the vocab projection.

Nothing here samples. The same tokens give the same logits.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .template import ChatTemplate, ToyChatTemplate

__all__ = [
    "FakeRunner",
    "LogitRow",
    "PrefixCache",
    "Readout",
    "Runner",
    "ToyTokenizer",
]


@runtime_checkable
class LogitRow(Protocol):
    """One vocab-sized row of logits. Anything indexable by token id works
    (a list, a numpy row, an ``mx.array`` row); readout only gathers a few ids."""

    def __getitem__(self, token_id: int, /) -> Any: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class PrefixCache:
    """A prefilled prefix. ``state`` is the runner's own KV object (opaque here)."""

    tokens: tuple[int, ...]
    state: Any = None

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class Readout:
    """Logits at the requested positions only, plus token accounting.

    ``input_tokens`` counts tokens actually prefilled by this call (the suffix
    when a prefix cache was used); ``cached_tokens`` counts those served from it.
    """

    positions: tuple[int, ...]
    logits: tuple[LogitRow, ...]
    input_tokens: int
    cached_tokens: int

    def at(self, position: int) -> LogitRow:
        try:
            return self.logits[self.positions.index(position)]
        except ValueError:
            raise KeyError(f"no logits at position {position}; have {self.positions}") from None


class Runner(Protocol):
    """The model seam. See the module docstring for the contract."""

    @property
    def chat_template(self) -> ChatTemplate: ...

    def tokenize(self, text: str) -> list[int]: ...

    def cache_prefix(self, tokens: Sequence[int]) -> PrefixCache: ...

    def prefill(
        self,
        tokens: Sequence[int],
        prefix_cache: PrefixCache | None = None,
        readout_positions: Sequence[int] | None = None,
    ) -> Readout: ...


def check_prefill_args(
    tokens: Sequence[int],
    prefix_cache: PrefixCache | None,
    readout_positions: Sequence[int] | None,
) -> tuple[int, tuple[int, ...]]:
    """Validate a ``prefill`` call; return ``(cached_len, positions)``.

    Shared so every Runner implementation enforces the same contract.
    """
    tokens = tuple(tokens)
    n_cached = 0
    if prefix_cache is not None:
        n_cached = len(prefix_cache)
        if tokens[:n_cached] != prefix_cache.tokens:
            raise ValueError("tokens do not start with the prefix cache's tokens")
        if len(tokens) == n_cached:
            raise ValueError("nothing to prefill: tokens equal the cached prefix")
    if not tokens:
        raise ValueError("cannot prefill an empty sequence")
    positions = (len(tokens) - 1,) if readout_positions is None else tuple(readout_positions)
    if not positions:
        raise ValueError("readout_positions is empty")
    for p in positions:
        if not n_cached <= p < len(tokens):
            raise ValueError(
                f"readout position {p} outside the prefilled suffix [{n_cached}, {len(tokens)})"
                " (a prefix cache holds KV, not logits)"
            )
    return n_cached, positions


# --- fake runner ----------------------------------------------------------------------------


class ToyTokenizer:
    """A deterministic, dependency-free tokenizer for tests.

    Pieces are special tokens (``<|name|>``), words with an optional leading
    space, single digits, single punctuation marks, or whitespace runs. Words
    longer than ``chunk`` letters split into ``chunk``-letter pieces, so long
    words are multi-token (``" billing"`` -> ``" bill"``, ``"ing"``) the way real
    subword tokenizers behave. Ids are a stable hash of the piece, so they are
    identical across processes and instances.
    """

    SPECIALS = ("<|pad|>", "<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
    _PIECE = re.compile(r"<\|[a-z]+\|>| ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]|\s+")

    def __init__(self, vocab_size: int = 50_000, chunk: int = 4):
        self.vocab_size = vocab_size
        self.chunk = chunk
        self._special_ids = {s: i for i, s in enumerate(self.SPECIALS)}
        self._seen: dict[int, str] = {i: s for s, i in self._special_ids.items()}

    def pieces(self, text: str) -> list[str]:
        out: list[str] = []
        for m in self._PIECE.finditer(text):
            piece = m.group()
            body = piece.lstrip(" ")
            if body.isalpha() and len(body) > self.chunk:
                lead = piece[: len(piece) - len(body)]
                chunks = [body[i : i + self.chunk] for i in range(0, len(body), self.chunk)]
                out.append(lead + chunks[0])
                out.extend(chunks[1:])
            else:
                out.append(piece)
        if "".join(out) != text:
            raise ValueError(f"ToyTokenizer cannot represent {text!r}")
        return out

    def piece_id(self, piece: str) -> int:
        if piece in self._special_ids:
            return self._special_ids[piece]
        n = len(self.SPECIALS)
        tid = n + zlib.crc32(piece.encode("utf-8")) % (self.vocab_size - n)
        self._seen.setdefault(tid, piece)
        return tid

    def encode(self, text: str) -> list[int]:
        return [self.piece_id(p) for p in self.pieces(text)]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._seen.get(i, f"<?{i}>") for i in ids)


class _FakeLogitRow:
    """A lazy vocab row: logit(token) computed on demand from the context digest."""

    __slots__ = ("_bias", "_digest", "_fn", "_lse", "_vocab")

    def __init__(self, digest: bytes, vocab: int, fn, bias: Mapping[int, float]):
        self._digest, self._vocab, self._fn, self._bias = digest, vocab, fn, bias
        self._lse: float | None = None

    def logsumexp(self) -> float:
        """Log-normaliser over the whole vocab (computed once, then cached)."""
        if self._lse is None:
            values = [self[i] for i in range(self._vocab)]
            m = max(values)
            self._lse = m + math.log(math.fsum(math.exp(v - m) for v in values))
        return self._lse

    def __len__(self) -> int:
        return self._vocab

    def __getitem__(self, token_id: int) -> float:
        if not 0 <= token_id < self._vocab:
            raise IndexError(token_id)
        return self._fn(self._digest, token_id) + self._bias.get(token_id, 0.0)


def _default_logit(digest: bytes, token_id: int) -> float:
    """Pseudo-random logit in [-4, 4), a pure function of (context, token)."""
    h = hashlib.blake2b(digest + struct.pack("<I", token_id), digest_size=8).digest()
    return (int.from_bytes(h, "little") / 2**64) * 8.0 - 4.0


@dataclass
class FakeRunner:
    """A deterministic :class:`Runner` with no weights.

    The logit for token ``v`` at position ``p`` is a pure function of
    ``tokens[: p + 1]`` and ``v`` (plus the static ``logit_bias``), so cached and
    uncached prefills agree and repeated calls are identical. ``logit_fn`` swaps
    the base function; it receives a digest of the context and the token id.

    ``projected_rows`` counts vocab projections performed, so tests can assert
    logits were computed only at readout positions.
    """

    tokenizer: ToyTokenizer = field(default_factory=ToyTokenizer)
    chat_template: ChatTemplate = field(default_factory=ToyChatTemplate)
    model_id: str = "fake"
    logit_bias: dict[int, float] = field(default_factory=dict)
    logit_fn: Callable[[bytes, int], float] = _default_logit
    projected_rows: int = 0
    prefilled_tokens: int = 0

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def cache_prefix(self, tokens: Sequence[int]) -> PrefixCache:
        tokens = tuple(tokens)
        self.prefilled_tokens += len(tokens)
        return PrefixCache(tokens=tokens, state=("fake-kv", len(tokens)))

    def prefill(
        self,
        tokens: Sequence[int],
        prefix_cache: PrefixCache | None = None,
        readout_positions: Sequence[int] | None = None,
    ) -> Readout:
        tokens = tuple(tokens)
        n_cached, positions = check_prefill_args(tokens, prefix_cache, readout_positions)
        self.prefilled_tokens += len(tokens) - n_cached
        rows = []
        for p in positions:
            digest = hashlib.blake2b(
                struct.pack(f"<{p + 1}I", *tokens[: p + 1]), digest_size=16
            ).digest()
            rows.append(_FakeLogitRow(digest, self.vocab_size, self.logit_fn, self.logit_bias))
            self.projected_rows += 1
        return Readout(
            positions=positions,
            logits=tuple(rows),
            input_tokens=len(tokens) - n_cached,
            cached_tokens=n_cached,
        )
