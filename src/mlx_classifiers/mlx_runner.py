"""The MLX-LM :class:`~mlx_classifiers.runner.Runner` (story 2.2), Gemma 4 E2B first.

``mlx`` and ``mlx_lm`` are imported lazily, so the package stays importable on
machines without them (``pip install mlx-classifiers[mlx]``).

The split prefill (the port seam the Swift repo mirrors)
---------------------------------------------------------
Given ``tokens``, an optional prefix cache, and ``readout_positions`` (absolute
indices, all in the uncached suffix), let ``p0 = min(positions)`` and
``p1 = max(positions)``:

1. **Cache-only rows** ``[n_cached, p0)``: run the backbone and evaluate only
   the KV cache state (``mx.eval([c.state for c in cache])``), never its output.
   MLX's lazy graph then drops everything the cache does not depend on: on
   Gemma 4 E2B that is the 20 KV-consumer layers (15..34 reuse layers 0..14's
   K/V) and the vocab projection. Chunked at ``prefill_step`` rows.
2. **Readout rows** ``[p0, p1]``: one forward through every layer, then gather
   the hidden rows at the readout positions and project *only those* through
   the (tied) lm_head and the final logit soft-cap. Rows after ``p1`` are never
   run. Never ``model(all)[:, -1]``.

Single readout (first-token mode) is therefore the cache-only prefill plus one
L=1 pass; sequence mode (readouts at every option token, contiguous) is exact
and pays the full layers only for those rows. The result is exact up to bf16
noise against a full forward (story 2.5 findings).

A prefix cache is the mlx-lm cache list after step 1 over the prefix tokens.
``prefill`` deep-copies it before extending, so a shared prefix is never
mutated (about 0.4 ms on E2B). Gemma's sliding layers use ``RotatingKVCache``
(window 512); prefixes longer than the window are tested.

Delta prefix caching (story 3.4)
--------------------------------
``prompt_cache`` (a :class:`~mlx_classifiers.prompt_cache.PromptCacheStore`, on
by default; None turns it off) keeps the KV state after step 1 of every
prefill, keyed by ``tokens[:p0]``. The next ``prefill`` looks up the stored
sequence sharing the longest common token prefix with its own tokens (capped
at ``p0``), rewinds a copy to that length (:func:`rewind_cache`), and runs
step 1 over the delta only. ``cached_tokens`` reports the hit length. A
declared prefix cache still works and is used when it is the longer hit.

Rewinding per layer: a ``KVCache`` trims to any length. A ``RotatingKVCache``
(sliding window ``W``) is first put in temporal order; if it never dropped a
row it is sliced to the hit length. Once it has slid, it holds rows
``[start, offset)`` and the next step only needs the last ``W - 1`` of them, so
a hit ``h`` is representable iff ``h - start >= W - 1``: the rewound cache is
those ``W - 1`` rows behind one placeholder row (``_idx = W``), which both of
mlx-lm's update paths drop or overwrite before attending. Otherwise that entry
cannot serve ``h`` and the store tries the next one (then no hit). In practice
a stored entry can rewind into its own last delta (step 1 keeps ``W - 1`` rows
plus the chunk just added; step 1 runs its remainder chunk first so that the
last chunk is a full ``prefill_step``), and an entry built from an empty cache
in one chunk rewinds anywhere.

MoE expert kernel (story 3.6): ``MLXRunner.load`` swaps the ``SwitchGLU`` of a
4-bit MoE (Gemma 4 26B-A4B) to :class:`~mlx_classifiers.moe.SegmentedSwitchGLU`,
which runs prefill chunks of 4 to 16 rows per expert (64 to 256 tokens on the
26B) through one-expert-per-threadgroup 16-row NAX tiles instead of MLX's
64-row tiles; bit-identical, and a no-op on dense models or without a NAX GPU.
``moe_kernel=False`` keeps the stock path. ``moe_kernel_layers`` counts the swaps.

INT8 (story 2.3): ``MLXRunner.load(repo, int8=True)`` swaps every decoder
layer's MLP and attention projections for the vendored W8A8 kernel
(:mod:`mlx_classifiers.int8`, Apple M5 / Metal 4 only). Calls under 64 rows
fall back to the stock linear inside the wrapper, so the L=1 readout pass is
stock either way; the cache-only prefill is where INT8 engages.

Images (story 6.3)
------------------
``prefill(..., images=slots)`` / ``cache_prefix(..., images=slots)`` take the
:class:`~mlx_classifiers.images.ImageSlot` runs compilation recorded. Any row
range that overlaps an image runs on **input embeddings** instead of token ids:
``embed_tokens(ids) * embed_scale`` with the image's projected soft tokens
(:mod:`mlx_classifiers.vision`, loaded on the first image, memoised per PNG
sha256) in its rows, per-layer inputs from the ids with image rows set to the
pad id, then the same decoder layers (:func:`~mlx_classifiers.vision.merge.forward_embeddings`,
mlx-lm's loop), so the cache-only / readout split, the KV sharing, the INT8
and MoE patches are all unchanged. Rows without an image take the token-id path
exactly as before. Only images that overlap the uncached rows are encoded.
On a model whose text config says ``use_bidirectional_attention: "vision"``
(the 26B-A4B), each image block attends to itself both ways, so cache-only
chunk boundaries are moved off image runs. The delta store keys on
:func:`~mlx_classifiers.images.keyed_tokens` (image rows carry the image's
hash), and a hit is never left inside an image run. ``vision_encode_ms``
records each encode.

MoE expert INT8 (story 2.8): on a 4-bit affine MoE (the 26B), ``int8=True`` also
swaps every ``SwitchGLU`` for :class:`~mlx_classifiers.int8.experts.Int8SwitchGLU`
(W4A8: the checkpoint's nibbles as the int4 operand, activations int8 per token).
It replaces the stock expert weights, so it runs at every row count (the L=1
readout included) and the story 3.6 kernel, which needs the stock encoding, is
not applied (``moe_kernel_layers`` is 0). ``int8_expert_layers`` counts the swaps;
``int8_experts=False`` keeps stock experts (with the 3.6 kernel) under ``int8``.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .images import ImageSlot, ImageTokens, image_tokens_from_config, keyed_tokens
from .prompt_cache import PromptCacheStore
from .runner import LogitRow, PrefixCache, Readout, check_images, check_prefill_args
from .template import ChatTemplate, HFChatTemplate

__all__ = ["DEFAULT_MODEL", "MLXRunner", "cache_nbytes", "rewind_cache"]

DEFAULT_MODEL = "mlx-community/gemma-4-E2B-it-qat-4bit"


def _mx_row(row: Any, lse: Any, vocab: int) -> LogitRow:
    """Wrap one evaluated float32 ``mx.array`` row (and its logsumexp) as a LogitRow."""
    return LogitRow(lambda t: row[t].item(), vocab, lse.item)


def _text_model(model: Any) -> Any:
    """The causal-LM module holding ``.model`` (backbone) and the head.

    Multimodal wrappers (``gemma4.Model``) nest it as ``language_model``."""
    return getattr(model, "language_model", model)


def _rotating_rewind_plan(c: Any, n: int) -> bool:
    """Whether a ``RotatingKVCache`` can drop its last ``n`` tokens (see module doc)."""
    if c.keep or c.keys is None or n > c.offset:
        return False
    shape = c.keys.shape[2]
    # rows _temporal_order would return: the whole ring, or the filled part
    ordered_len = shape if (c._idx == shape or c._idx < c.offset) else c._idx
    start = c.offset - ordered_len
    return start == 0 or (c.offset - n) - start >= c.max_size - 1


def _rotating_rewind(c: Any, n: int) -> None:
    """Rewind a (copied) ``RotatingKVCache`` by ``n`` tokens in place."""
    import mlx.core as mx

    keys, values = c._temporal_order(c.keys), c._temporal_order(c.values)
    start = c.offset - keys.shape[2]
    h = c.offset - n
    if start == 0:
        keys, values = keys[..., :h, :], values[..., :h, :]
        c.keys, c.values, c.offset, c._idx = keys, values, h, h
        return
    w = c.max_size
    lo, hi = h - start - (w - 1), h - start
    pad_k = mx.zeros((*keys.shape[:2], 1, keys.shape[3]), keys.dtype)
    pad_v = mx.zeros((*values.shape[:2], 1, values.shape[3]), values.dtype)
    c.keys = mx.concatenate([pad_k, keys[..., lo:hi, :]], axis=2)
    c.values = mx.concatenate([pad_v, values[..., lo:hi, :]], axis=2)
    c.offset, c._idx = h, w


def rewind_cache(cache: list[Any], stored_len: int, hit_len: int) -> list[Any] | None:
    """A deep copy of the mlx-lm cache list ``cache`` (covering ``stored_len``
    tokens) holding only the first ``hit_len``, or None if some layer cannot
    represent it. ``cache`` itself is never mutated."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    n = stored_len - hit_len
    if n < 0:
        raise ValueError("hit_len beyond the stored sequence")
    for c in cache:
        if n == 0:
            break
        if isinstance(c, RotatingKVCache):
            if not _rotating_rewind_plan(c, n):
                return None
        elif isinstance(c, KVCache):
            if n > c.offset:
                return None
        elif not (hasattr(c, "trim") and getattr(c, "is_trimmable", lambda: False)()):
            return None
    out = copy.deepcopy(cache)
    if n:
        for c in out:
            if isinstance(c, RotatingKVCache):
                _rotating_rewind(c, n)
            else:
                c.trim(n)
    return out


def _bounds_off_runs(bounds: list[int], runs: Sequence[tuple[int, int]]) -> list[int]:
    """Chunk boundaries with any that fall strictly inside a run moved to its start
    (or its end, when the start is the previous boundary); duplicates dropped."""
    out: list[int] = []
    for b in bounds:
        for a, e in runs:
            if a < b < e:
                b = a if (not out or a > out[-1]) else e
                break
        if not out or b > out[-1]:
            out.append(b)
    return out


def cache_nbytes(cache: list[Any]) -> int:
    """Bytes held by an mlx-lm cache list (0 for layers that do not say)."""
    total = 0
    for c in cache:
        try:
            total += int(c.nbytes)
        except (AttributeError, NotImplementedError, TypeError):
            pass
    return total


@dataclass
class MLXRunner:
    """A Runner over an mlx-lm ``(model, tokenizer)`` pair. Build with :meth:`load`."""

    model: Any
    tokenizer: Any
    model_id: str = DEFAULT_MODEL
    prefill_step: int = 2048
    chat_template: ChatTemplate = field(default=None)  # type: ignore[assignment]
    #: True when the W8A8 INT8 kernels were patched in at load (story 2.3).
    int8: bool = False
    #: Shapes seen by the vocab projection, for tests/profiling: (rows, hidden).
    head_calls: list[tuple[int, ...]] = field(default_factory=list)
    #: Delta prefix cache (story 3.4); None turns it off.
    prompt_cache: PromptCacheStore | None = field(default_factory=PromptCacheStore)
    #: Each prefill as ``(cached_tokens, computed_tokens)``, for tests/profiling.
    prefill_calls: list[tuple[int, int]] = field(default_factory=list)
    #: MoE layers running the per-expert kernel (story 3.6); 0 for the stock path.
    moe_kernel_layers: int = 0
    #: MoE layers whose experts run the W4A8 kernel (story 2.8); 0 unless ``int8``.
    int8_expert_layers: int = 0
    #: The image placeholder convention (story 6.1); None: no vision path.
    image_tokens: ImageTokens | None = None
    #: Snapshot directory holding the vision weights, loaded on the first image.
    vision_dir: str | None = None
    #: Image blocks attend bidirectionally (text config ``use_bidirectional_attention``).
    vision_bidirectional: bool = False
    #: The loaded :class:`~mlx_classifiers.vision.VisionTower` (None until needed).
    vision: Any = None
    #: Milliseconds of each image encode (cache misses only), for tests/profiling.
    vision_encode_ms: list[float] = field(default_factory=list)
    #: Story 6.6: encode images on the fast path (compiled segments, shape-cached
    #: tables, no one-hot position matmul, no all-zero mask); bit-equal to the vendored
    #: path on E2B and the 26B. False runs the vendored path (the reference).
    vision_fast: bool = True

    def __post_init__(self) -> None:
        if self.chat_template is None:
            self.chat_template = HFChatTemplate(self.tokenizer)
        lm = _text_model(self.model)
        self._backbone = lm.model
        self._lm = lm

    @classmethod
    def load(
        cls,
        repo: str = DEFAULT_MODEL,
        *,
        int8: bool = False,
        moe_kernel: bool = True,
        int8_experts: bool = True,
        vision: bool = False,
        **kwargs: Any,
    ) -> MLXRunner:
        """Load ``repo`` with mlx-lm. ``int8=True`` then applies the W8A8 ``patch_mlps``
        and ``patch_projs`` (dequantise the checkpoint, requantise per output channel;
        it keeps the stock linears for the short-row fallback, so resident memory grows,
        about 3.4 to 5.2 GB on E2B). ``moe_kernel`` (default on) applies
        :func:`mlx_classifiers.moe.patch_model` (story 3.6; bit-identical). On a MoE,
        ``int8`` with ``int8_experts`` (default) also runs the experts W4A8 (story 2.8),
        which supersedes the 3.6 kernel on that model.

        A checkpoint with a vision tower gets ``image_tokens`` from its JSON
        (story 6.1); the vision weights load on the first image, or now with
        ``vision=True`` (story 6.3)."""
        import json

        from mlx_lm import load
        from mlx_lm.utils import _download

        model, tokenizer = load(repo)
        path = _download(repo)
        image_tokens = image_tokens_from_config(path)
        text_config = json.loads((path / "config.json").read_text()).get("text_config") or {}
        expert_layers = 0
        if int8:
            from .int8 import patch_experts, patch_model

            patch_model(model)
            if int8_experts:
                expert_layers = patch_experts(model)
        layers = 0
        if moe_kernel:
            from .moe import patch_model as patch_moe

            layers = patch_moe(model)
        runner = cls(
            model=model,
            tokenizer=tokenizer,
            model_id=repo,
            int8=int8,
            moe_kernel_layers=layers,
            int8_expert_layers=expert_layers,
            image_tokens=image_tokens,
            vision_dir=str(path) if image_tokens is not None else None,
            vision_bidirectional=text_config.get("use_bidirectional_attention") == "vision",
            **kwargs,
        )
        if vision:
            runner.load_vision()
        return runner

    def load_vision(self) -> Any:
        """The :class:`~mlx_classifiers.vision.VisionTower`, loading it on first use."""
        if self.vision is None:
            if self.vision_dir is None:
                raise ValueError(f"{self.model_id} has no vision tower")
            from .vision import load_vision_tower

            hidden = self._backbone.config.hidden_size
            self.vision = load_vision_tower(
                self.vision_dir, text_hidden_size=hidden, fast=self.vision_fast
            )
        return self.vision

    # --- Runner protocol ---------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def cache_prefix(
        self, tokens: Sequence[int], *, delta: bool = True, images: Sequence[ImageSlot] = ()
    ) -> PrefixCache:
        tokens = tuple(tokens)
        slots = self._check_images(tokens, images)
        key = keyed_tokens(tokens, slots)
        n_cached, cache = self._start(key, len(tokens), None, delta, slots)
        self._cache_only(tokens[n_cached:], cache, slots, n_cached)
        self._store(key, cache, n_cached, delta)
        return PrefixCache(tokens=tokens, state=cache, images=slots)

    def prefill(
        self,
        tokens: Sequence[int],
        prefix_cache: PrefixCache | None = None,
        readout_positions: Sequence[int] | None = None,
        *,
        delta: bool = True,
        images: Sequence[ImageSlot] = (),
    ) -> Readout:
        import mlx.core as mx

        tokens = tuple(tokens)
        _, positions = check_prefill_args(tokens, prefix_cache, readout_positions, images)
        slots = self._check_images(tokens, images)
        p0, p1 = min(positions), max(positions)
        if any(s.start < p0 < s.end or p0 <= s.start <= p1 for s in slots):
            raise ValueError("readout positions must not fall inside or before an image run")
        key = keyed_tokens(tokens, slots)
        n_cached, cache = self._start(key, p0, prefix_cache, delta, slots)
        self.prefill_calls.append((n_cached, p1 + 1 - n_cached))

        self._cache_only(tokens[n_cached:p0], cache, slots, n_cached)
        self._store(key[:p0], cache, n_cached, delta)
        hidden = self._rows(tokens, p0, p1 + 1, cache, slots)
        rows = mx.array([p - p0 for p in positions])
        logits = self._head(hidden[:, rows, :])[0].astype(mx.float32)  # (k, vocab)
        lse = mx.logsumexp(logits, axis=-1)
        mx.eval(logits, lse)

        vocab = logits.shape[-1]
        out = tuple(_mx_row(logits[i], lse[i], vocab) for i in range(len(positions)))
        return Readout(
            positions=positions,
            logits=out,
            input_tokens=p1 + 1 - n_cached,
            cached_tokens=n_cached,
        )

    # --- internals ---------------------------------------------------------------------

    def _check_images(
        self, tokens: tuple[int, ...], images: Sequence[ImageSlot]
    ) -> tuple[ImageSlot, ...]:
        slots = check_images(tokens, images)
        if not slots:
            return slots
        if self.image_tokens is None:
            raise ValueError(f"{self.model_id} has no vision path")
        image_id = self.image_tokens.image_token_id
        for s in slots:
            if not s.data:
                raise ValueError(f"image slot at {s.start} carries no PNG bytes")
            if any(t != image_id for t in tokens[s.start : s.end]):
                raise ValueError(f"image run at {s.start} is not all image-token ids")
        return slots

    def _start(
        self,
        key: tuple[int, ...],
        max_len: int,
        prefix_cache: PrefixCache | None,
        delta: bool,
        slots: Sequence[ImageSlot] = (),
    ) -> tuple[int, list[Any]]:
        """``(n_cached, working cache)``: the longer of the declared prefix and the
        delta store's hit (both copies), else an empty cache. ``key`` is the
        image-keyed token sequence; a hit is never left inside an image run."""
        from mlx_lm.models.cache import make_prompt_cache

        have = len(prefix_cache) if prefix_cache is not None else 0
        if delta and self.prompt_cache is not None:
            hit, state = self.prompt_cache.lookup(key, rewind_cache, max_len=max_len, min_len=have)
            inside = next((s for s in slots if s.start < hit < s.end), None)
            if state is not None and inside is not None:
                hit, state = self.prompt_cache.lookup(
                    key, rewind_cache, max_len=inside.start, min_len=have
                )
            if state is not None:
                return hit, state
        if prefix_cache is not None:
            return have, copy.deepcopy(prefix_cache.state)
        return 0, make_prompt_cache(self.model)

    def _store(self, key: tuple[int, ...], cache: list[Any], n_cached: int, delta: bool) -> None:
        """Keep a copy of ``cache`` (covering ``key``) when this call added rows."""
        if delta and self.prompt_cache is not None and len(key) > n_cached:
            snapshot = copy.deepcopy(cache)
            self.prompt_cache.insert(key, snapshot, cache_nbytes(snapshot))

    def _cache_only(
        self,
        tokens: Sequence[int],
        cache: list[Any],
        slots: Sequence[ImageSlot] = (),
        base: int = 0,
    ) -> None:
        """Advance ``cache`` over ``tokens`` (the prompt's rows ``base ..``)
        evaluating only the KV state.

        Chunks of ``prefill_step`` rows, the remainder *first*: a sliding layer
        keeps ``W - 1`` rows plus the last chunk, which is how far a stored entry
        can later be rewound (story 3.4), so the last chunk is the full one.
        ``slots`` are image runs in absolute positions; on a bidirectional-vision
        model no chunk boundary falls inside one."""
        import mlx.core as mx

        n, step = len(tokens), self.prefill_step
        if n == 0:
            return
        first = n % step or step
        bounds = [0, *range(first, n, step), n]
        if slots and self.vision_bidirectional:
            bounds = _bounds_off_runs(bounds, [(s.start - base, s.end - base) for s in slots])
        local = [s for s in slots if s.end > base and s.start < base + n]
        for start, end in itertools.pairwise(bounds):
            if local:
                self._rows(tokens, start, end, cache, local, base)
            else:
                chunk = mx.array(tokens[start:end])[None]
                self._backbone(chunk, cache=cache)  # output deliberately never evaluated
            mx.eval([c.state for c in cache])

    def _image_features(self, slot: ImageSlot) -> Any:
        """``[length, hidden]`` soft tokens of ``slot`` (encoded once per sha256)."""
        import time

        tower = self.load_vision()
        before = tower.encodes
        t0 = time.perf_counter()
        feats = tower.encode_png(slot.data, slot.sha256)
        if tower.encodes != before:
            self.vision_encode_ms.append((time.perf_counter() - t0) * 1e3)
        if feats.shape[0] != slot.length:
            raise ValueError(
                f"image encodes to {feats.shape[0]} soft tokens, run has {slot.length}"
            )
        return feats

    def _rows(
        self,
        tokens: Sequence[int],
        start: int,
        end: int,
        cache: list[Any],
        slots: Sequence[ImageSlot],
        base: int = 0,
    ) -> Any:
        """The backbone over ``tokens[start:end]`` (prompt rows ``base + start ..``):
        token ids when no image overlaps, else merged input embeddings."""
        import mlx.core as mx

        ids = mx.array(tokens[start:end])
        lo, hi = base + start, base + end
        pieces = []
        for s in slots:
            a, b = max(s.start, lo), min(s.end, hi)
            if a < b:
                feats = self._image_features(s)[a - s.start : b - s.start]
                pieces.append((a - lo, b - a, feats))
        if not pieces:
            return self._backbone(ids[None], cache=cache)
        from .vision.merge import (
            block_ids,
            forward_embeddings,
            merged_embeddings,
            per_layer_token_ids,
        )

        h = merged_embeddings(self._backbone, ids, pieces)
        blocks = block_ids(end - start, pieces) if self.vision_bidirectional else None
        pl = per_layer_token_ids(ids, pieces)
        return forward_embeddings(self._backbone, h, cache, pl, blocks)

    def _head(self, hidden: Any) -> Any:
        """Vocab projection + final soft-cap, mirroring the model's own ``__call__``."""
        import mlx.core as mx

        self.head_calls.append(tuple(hidden.shape))
        lm = self._lm
        if hasattr(lm, "lm_head"):
            out = lm.lm_head(hidden)
        else:  # tied embeddings (Gemma)
            out = lm.model.embed_tokens.as_linear(hidden)
        cap = getattr(lm, "final_logit_softcapping", None)
        if cap is not None:
            out = mx.tanh(out / cap) * cap
        return out
