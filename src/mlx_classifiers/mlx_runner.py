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
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .runner import LogitRow, PrefixCache, Readout, check_prefill_args
from .template import ChatTemplate, HFChatTemplate

__all__ = ["DEFAULT_MODEL", "MLXRunner"]

DEFAULT_MODEL = "mlx-community/gemma-4-E2B-it-qat-4bit"


def _mx_row(row: Any, lse: Any, vocab: int) -> LogitRow:
    """Wrap one evaluated float32 ``mx.array`` row (and its logsumexp) as a LogitRow."""
    return LogitRow(lambda t: row[t].item(), vocab, lse.item)


def _text_model(model: Any) -> Any:
    """The causal-LM module holding ``.model`` (backbone) and the head.

    Multimodal wrappers (``gemma4.Model``) nest it as ``language_model``."""
    return getattr(model, "language_model", model)


@dataclass
class MLXRunner:
    """A Runner over an mlx-lm ``(model, tokenizer)`` pair. Build with :meth:`load`."""

    model: Any
    tokenizer: Any
    model_id: str = DEFAULT_MODEL
    prefill_step: int = 2048
    chat_template: ChatTemplate = field(default=None)  # type: ignore[assignment]
    #: Shapes seen by the vocab projection, for tests/profiling: (rows, hidden).
    head_calls: list[tuple[int, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.chat_template is None:
            self.chat_template = HFChatTemplate(self.tokenizer)
        lm = _text_model(self.model)
        self._backbone = lm.model
        self._lm = lm

    @classmethod
    def load(cls, repo: str = DEFAULT_MODEL, **kwargs: Any) -> MLXRunner:
        from mlx_lm import load

        model, tokenizer = load(repo)
        return cls(model=model, tokenizer=tokenizer, model_id=repo, **kwargs)

    # --- Runner protocol ---------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def cache_prefix(self, tokens: Sequence[int]) -> PrefixCache:
        from mlx_lm.models.cache import make_prompt_cache

        tokens = tuple(tokens)
        cache = make_prompt_cache(self.model)
        self._cache_only(tokens, cache)
        return PrefixCache(tokens=tokens, state=cache)

    def prefill(
        self,
        tokens: Sequence[int],
        prefix_cache: PrefixCache | None = None,
        readout_positions: Sequence[int] | None = None,
    ) -> Readout:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        tokens = tuple(tokens)
        n_cached, positions = check_prefill_args(tokens, prefix_cache, readout_positions)
        if prefix_cache is not None:
            cache = copy.deepcopy(prefix_cache.state)
        else:
            cache = make_prompt_cache(self.model)
        p0, p1 = min(positions), max(positions)

        self._cache_only(tokens[n_cached:p0], cache)
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

    def _cache_only(self, tokens: Sequence[int], cache: list[Any]) -> None:
        """Advance ``cache`` over ``tokens`` evaluating only the KV state."""
        import mlx.core as mx

        for start in range(0, len(tokens), self.prefill_step):
            chunk = mx.array(tokens[start : start + self.prefill_step])[None]
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
