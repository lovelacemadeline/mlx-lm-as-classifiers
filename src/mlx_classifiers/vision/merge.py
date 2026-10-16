# The merge rule follows mlx-vlm 0.7.8 (PyPI wheel; https://github.com/Blaizzy/mlx-vlm), MIT
# licence, Copyright (c) 2025 Prince Canuma -- full text in LICENSE.mlx-vlm beside this file:
#   mlx_vlm/models/gemma4/gemma4.py  sha256 c0b57f2fd9eac5b2185b4aa081606f4cb85607c1d59eaef7781931af48194b25
#   (Model.get_input_embeddings: text rows embed_tokens(ids) * embed_scale, image rows the
#    projected soft tokens in order; per-layer inputs from the ids with every multimodal
#    position set to 0, the pad id)
# and the bidirectional image-block mask overlay of
#   mlx_vlm/models/gemma4/language.py  sha256 c15718b60a0ad6f65aefd1bd6df149469d2d09d9b0f1f534d4f750cc19367ec4
#   (_block_sequence_ids_for_mask, _apply_blockwise_bidirectional_overlay; applied when the
#    text config says use_bidirectional_attention == "vision", as the 26B-A4B does)
#
# forward_embeddings is the layer loop of mlx-lm 0.31.3 (MIT, Copyright (c) 2023 Apple Inc.)
#   mlx_lm/models/gemma4_text.py  sha256 77f46bc3f162a0b9513157dade4be2c381d4df3295262c69034a53d46111370f
#   (Gemma4TextModel.__call__ from "Get the extra inputs per layer" on), taking the already
#   scaled, merged embeddings instead of token ids, and optional mask overrides.
"""Merge image soft tokens into the text embeddings, then run the decoder layers on them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

__all__ = ["block_ids", "forward_embeddings", "merged_embeddings", "per_layer_token_ids"]

#: Image runs inside a segment: ``(offset in segment, length, features [length, D])``.
Pieces = Sequence[tuple[int, int, Any]]


def merged_embeddings(backbone: Any, ids: mx.array, pieces: Pieces) -> mx.array:
    """``[1, L, D]``: ``embed_tokens(ids) * embed_scale`` with each image run's rows
    replaced by its (projected) soft tokens, cast to the embedding dtype."""
    text = backbone.embed_tokens(ids[None]) * backbone.embed_scale
    if not pieces:
        return text
    parts, at = [], 0
    for off, n, feats in sorted(pieces, key=lambda p: p[0]):
        if off > at:
            parts.append(text[:, at:off])
        parts.append(feats.astype(text.dtype)[None])
        at = off + n
    if at < text.shape[1]:
        parts.append(text[:, at:])
    return mx.concatenate(parts, axis=1)


def per_layer_token_ids(ids: mx.array, pieces: Pieces) -> mx.array:
    """``ids`` with every image position set to 0 (the pad id), as Gemma 4 does."""
    if not pieces:
        return ids
    mask = mx.zeros(ids.shape, dtype=mx.bool_)
    pos = mx.arange(ids.shape[0])
    for off, n, _ in pieces:
        mask = mask | ((pos >= off) & (pos < off + n))
    return mx.where(mask, mx.zeros_like(ids), ids)


def block_ids(length: int, pieces: Pieces) -> mx.array:
    """``[length]`` int32: image ``k``'s rows hold ``k``, text rows ``-1``."""
    out = [-1] * length
    for k, (off, n, _) in enumerate(sorted(pieces, key=lambda p: p[0])):
        out[off : off + n] = [k] * n
    return mx.array(out, dtype=mx.int32)


def _overlay(mask: Any, blocks: mx.array) -> mx.array:
    """``mask`` (bool ``[N, K]``, last ``N`` columns = this segment) with every pair of
    rows in the same image block allowed to attend both ways."""
    n = blocks.shape[0]
    same = (blocks[:, None] != -1) & (blocks[:, None] == blocks[None, :])
    k = mask.shape[-1]
    if k == n:
        return mask | same
    return mx.concatenate([mask[..., : k - n], mask[..., k - n :] | same], axis=-1)


def _make_masks(backbone: Any, h: mx.array, cache: list[Any], blocks: mx.array | None):
    """mlx-lm's per-layer-type masks, as arrays with the block overlay when ``blocks``."""
    from mlx_lm.models.base import create_attention_mask

    masks: dict[str, Any] = {}
    out = []
    for layer, c in zip(backbone.layers, cache, strict=True):
        kind = layer.layer_type
        if kind not in masks:
            window = backbone.window_size if kind == "sliding_attention" else None
            as_array = blocks is not None and h.shape[1] > 1
            m = create_attention_mask(h, c, window_size=window, return_array=as_array)
            if as_array:
                m = _overlay(m, blocks)
            masks[kind] = m
        out.append(masks[kind])
    return out


def forward_embeddings(
    backbone: Any,
    h: mx.array,
    cache: list[Any],
    per_layer_ids: mx.array | None,
    blocks: mx.array | None = None,
) -> mx.array:
    """The Gemma 4 text backbone over scaled input embeddings ``h`` ``[1, L, D]``.

    ``per_layer_ids`` (``[L]``) feed the per-layer embeddings on models that have
    them (E2B); ``blocks`` (from :func:`block_ids`) switches on the bidirectional
    image-block mask. Mirrors ``Gemma4TextModel.__call__`` otherwise, so the KV
    cache it fills is the one the token-id path would fill."""
    if backbone.hidden_size_per_layer_input:
        raw = backbone._get_per_layer_inputs(per_layer_ids[None])
        per_layer_inputs = backbone._project_per_layer_inputs(h, raw)
        per_layer_inputs = [per_layer_inputs[:, :, i, :] for i, _ in enumerate(backbone.layers)]
    else:
        per_layer_inputs = [None] * len(backbone.layers)

    cache = cache + [None] * (len(backbone.layers) - len(cache))
    masks = _make_masks(backbone, h, cache, blocks)
    intermediates = [(None, None)] * len(backbone.layers)
    for idx, (layer, c, mask, prev_idx, per_layer_input) in enumerate(
        zip(backbone.layers, cache, masks, backbone.previous_kvs, per_layer_inputs, strict=True)
    ):
        kvs, offset = intermediates[prev_idx]
        h, kvs, offset = layer(
            h, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset
        )
        intermediates[idx] = (kvs, offset)
    return backbone.norm(h)
