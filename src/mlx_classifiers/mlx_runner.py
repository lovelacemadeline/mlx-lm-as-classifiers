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
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .prompt_cache import PromptCacheStore
from .runner import LogitRow, PrefixCache, Readout, check_prefill_args
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
        **kwargs: Any,
    ) -> MLXRunner:
        """Load ``repo`` with mlx-lm. ``int8=True`` then applies the W8A8 ``patch_mlps``
        and ``patch_projs`` (dequantise the checkpoint, requantise per output channel;
        it keeps the stock linears for the short-row fallback, so resident memory grows,
        about 3.4 to 5.2 GB on E2B). ``moe_kernel`` (default on) applies
        :func:`mlx_classifiers.moe.patch_model` (story 3.6; bit-identical)."""
        from mlx_lm import load

        model, tokenizer = load(repo)
        if int8:
            from .int8 import patch_model

            patch_model(model)
        layers = 0
        if moe_kernel:
            from .moe import patch_model as patch_moe

            layers = patch_moe(model)
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=repo,
            int8=int8,
            moe_kernel_layers=layers,
            **kwargs,
        )

    # --- Runner protocol ---------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def cache_prefix(self, tokens: Sequence[int], *, delta: bool = True) -> PrefixCache:
        tokens = tuple(tokens)
        n_cached, cache = self._start(tokens, len(tokens), None, delta)
        self._cache_only(tokens[n_cached:], cache)
        self._store(tokens, cache, n_cached, delta)
        return PrefixCache(tokens=tokens, state=cache)

    def prefill(
        self,
        tokens: Sequence[int],
        prefix_cache: PrefixCache | None = None,
        readout_positions: Sequence[int] | None = None,
        *,
        delta: bool = True,
    ) -> Readout:
        import mlx.core as mx

        tokens = tuple(tokens)
        _, positions = check_prefill_args(tokens, prefix_cache, readout_positions)
        p0, p1 = min(positions), max(positions)
        n_cached, cache = self._start(tokens, p0, prefix_cache, delta)
        self.prefill_calls.append((n_cached, p1 + 1 - n_cached))

        self._cache_only(tokens[n_cached:p0], cache)
        self._store(tokens[:p0], cache, n_cached, delta)
        hidden = self._backbone(mx.array(tokens[p0 : p1 + 1])[None], cache=cache)
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

    def _start(
        self, tokens: tuple[int, ...], max_len: int, prefix_cache: PrefixCache | None, delta: bool
    ) -> tuple[int, list[Any]]:
        """``(n_cached, working cache)``: the longer of the declared prefix and the
        delta store's hit (both copies), else an empty cache."""
        from mlx_lm.models.cache import make_prompt_cache

        have = len(prefix_cache) if prefix_cache is not None else 0
        if delta and self.prompt_cache is not None:
            hit, state = self.prompt_cache.lookup(
                tokens, rewind_cache, max_len=max_len, min_len=have
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

    def _cache_only(self, tokens: Sequence[int], cache: list[Any]) -> None:
        """Advance ``cache`` over ``tokens`` evaluating only the KV state.

        Chunks of ``prefill_step`` rows, the remainder *first*: a sliding layer
        keeps ``W - 1`` rows plus the last chunk, which is how far a stored entry
        can later be rewound (story 3.4), so the last chunk is the full one."""
        import mlx.core as mx

        n, step = len(tokens), self.prefill_step
        if n == 0:
            return
        first = n % step or step
        for start, end in [(0, first)] + [(a, a + step) for a in range(first, n, step)]:
            chunk = mx.array(tokens[start:end])[None]
            self._backbone(chunk, cache=cache)  # output deliberately never evaluated
            mx.eval([c.state for c in cache])

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
