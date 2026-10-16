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

Delta prefix caching (story 3.4) is an optional extension, not part of the
protocol: a runner with a ``prompt_cache`` attribute (a
:class:`~mlx_classifiers.prompt_cache.PromptCacheStore`, or None when off)
accepts ``delta: bool = True`` on ``prefill`` and ``cache_prefix``. With it, the
call also reuses the longest cached token prefix of ``tokens`` (when longer
than ``prefix_cache``), prefills only the rest, and stores its own state for
later calls. ``Readout.cached_tokens`` is then that hit length.

Images (story 6.1/6.3) are an optional extension too: a runner with a vision
path has an ``image_tokens`` attribute (an
:class:`~mlx_classifiers.images.ImageTokens`, the placeholder convention prompt
compilation uses) and accepts ``images: Sequence[ImageSlot] = ()`` on
``prefill`` and ``cache_prefix``: each slot names the run of placeholder ids in
``tokens`` that the image's soft tokens replace. A prefix cache built with
images remembers them (``PrefixCache.images``), and the delta cache keys on
:func:`~mlx_classifiers.images.keyed_tokens`, so the same placeholder ids with a
different image never share KV.

Nothing here samples. The same tokens give the same logits.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .images import ImageSlot, ImageTokens, keyed_tokens
from .prompt_cache import PromptCacheStore
from .template import ChatTemplate, ToyChatTemplate

__all__ = [
    "FakeRunner",
    "LogitRow",
    "PrefixCache",
    "Readout",
    "Runner",
    "ToyTokenizer",
]


class LogitRow:
    """One vocab-sized row of logits at a readout position; every Runner returns this.

    ``row[token_id]`` is a float logit; ``row.logsumexp()`` is the full-vocab
    log-normaliser (sequence mode needs it), computed once on first use. The
    backing store is the runner's business: the FakeRunner computes logits on
    demand, the MLX runner wraps an ``mx.array``. Readout gathers only a few ids.
    """

    __slots__ = ("_lookup", "_lse", "_lse_fn", "_size")

    def __init__(
        self,
        lookup: Callable[[int], float],
        size: int,
        logsumexp: Callable[[], float] | None = None,
    ):
        self._lookup, self._size, self._lse_fn = lookup, size, logsumexp
        self._lse: float | None = None

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, token_id: int) -> float:
        if not 0 <= token_id < self._size:
            raise IndexError(token_id)
        return float(self._lookup(token_id))

    def logsumexp(self) -> float:
        if self._lse is None:
            if self._lse_fn is not None:
                self._lse = float(self._lse_fn())
            else:
                values = [self[i] for i in range(self._size)]
                m = max(values)
                self._lse = m + math.log(math.fsum(math.exp(v - m) for v in values))
        return self._lse

    @classmethod
    def from_list(cls, values: Sequence[float]) -> LogitRow:
        values = list(values)
        return cls(values.__getitem__, len(values))


@dataclass(frozen=True)
class PrefixCache:
    """A prefilled prefix. ``state`` is the runner's own KV object (opaque here);
    ``images`` are the image slots inside ``tokens`` (story 6.3)."""

    tokens: tuple[int, ...]
    state: Any = None
    images: tuple[ImageSlot, ...] = ()

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


def check_images(tokens: Sequence[int], images: Sequence[ImageSlot]) -> tuple[ImageSlot, ...]:
    """Image slots sorted by position, non-overlapping and inside ``tokens``."""
    slots = tuple(sorted(images, key=lambda s: s.start))
    end = 0
    for s in slots:
        if s.length < 1 or s.start < end or s.end > len(tokens):
            raise ValueError(f"image run [{s.start}, {s.end}) overlaps or is outside the tokens")
        end = s.end
    return slots


def check_prefill_args(
    tokens: Sequence[int],
    prefix_cache: PrefixCache | None,
    readout_positions: Sequence[int] | None,
    images: Sequence[ImageSlot] = (),
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
        inside = tuple(s for s in check_images(tokens, images) if s.start < n_cached)
        if inside != tuple(prefix_cache.images) or any(s.end > n_cached for s in inside):
            raise ValueError("the prefix cache holds different images than these tokens")
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

    ``prompt_cache`` (default None: off) turns on delta prefix caching (story
    3.4); the fake KV is always rewindable. ``prefill_calls`` records each
    prefill as ``(cached_tokens, computed_tokens)``.
    """

    tokenizer: ToyTokenizer = field(default_factory=ToyTokenizer)
    chat_template: ChatTemplate = field(default_factory=ToyChatTemplate)
    model_id: str = "fake"
    logit_bias: dict[int, float] = field(default_factory=dict)
    logit_fn: Callable[[bytes, int], float] = _default_logit
    projected_rows: int = 0
    prefilled_tokens: int = 0
    prompt_cache: PromptCacheStore | None = None
    prefill_calls: list[tuple[int, int]] = field(default_factory=list)
    #: The toy image placeholder convention (story 6.1): Gemma's soft-token counts
    #: with ToyTokenizer markup. Image slots change the logits through their sha256.
    image_tokens: ImageTokens | None = None

    def __post_init__(self) -> None:
        if self.image_tokens is None:
            self.image_tokens = ImageTokens(
                boi="<|boi|>",
                image="<|image|>",
                eoi="<|eoi|>",
                image_token_id=self.tokenizer.piece_id("<|image|>"),
            )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def _lookup(self, digest: bytes) -> Callable[[int], float]:
        fn, bias = self.logit_fn, self.logit_bias
        return lambda token_id: fn(digest, token_id) + bias.get(token_id, 0.0)

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    @staticmethod
    def _rewind(state: Any, stored_len: int, hit_len: int) -> Any:
        return ("fake-kv", hit_len)

    def _delta_hit(self, tokens: tuple[int, ...], max_len: int, have: int, delta: bool) -> int:
        if not delta or self.prompt_cache is None:
            return have
        hit, _ = self.prompt_cache.lookup(tokens, self._rewind, max_len=max_len, min_len=have)
        return max(hit, have)

    def cache_prefix(
        self, tokens: Sequence[int], *, delta: bool = True, images: Sequence[ImageSlot] = ()
    ) -> PrefixCache:
        tokens = tuple(tokens)
        images = check_images(tokens, images)
        key = keyed_tokens(tokens, images)
        hit = self._delta_hit(key, len(key), 0, delta)
        self.prefilled_tokens += len(tokens) - hit
        state = ("fake-kv", len(tokens))
        if delta and self.prompt_cache is not None and len(tokens) > hit:
            self.prompt_cache.insert(key, state, len(tokens))
        return PrefixCache(tokens=tokens, state=state, images=images)

    def prefill(
        self,
        tokens: Sequence[int],
        prefix_cache: PrefixCache | None = None,
        readout_positions: Sequence[int] | None = None,
        *,
        delta: bool = True,
        images: Sequence[ImageSlot] = (),
    ) -> Readout:
        tokens = tuple(tokens)
        n_cached, positions = check_prefill_args(tokens, prefix_cache, readout_positions, images)
        key = keyed_tokens(tokens, check_images(tokens, images))
        p0 = min(positions)
        n_cached = self._delta_hit(key, p0, n_cached, delta)
        if delta and self.prompt_cache is not None and p0 > n_cached:
            self.prompt_cache.insert(key[:p0], ("fake-kv", p0), p0)
        self.prefilled_tokens += len(tokens) - n_cached
        self.prefill_calls.append((n_cached, len(tokens) - n_cached))
        rows = []
        for p in positions:
            ctx = key[: p + 1]
            packed = struct.pack(f"<{p + 1}I", *ctx) if not images else repr(ctx).encode()
            digest = hashlib.blake2b(packed, digest_size=16).digest()
            rows.append(LogitRow(self._lookup(digest), self.vocab_size))
            self.projected_rows += 1
        return Readout(
            positions=positions,
            logits=tuple(rows),
            input_tokens=len(tokens) - n_cached,
            cached_tokens=n_cached,
        )
