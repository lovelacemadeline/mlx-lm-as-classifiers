# Vendored from mlx-vlm 0.7.8 (PyPI wheel; https://github.com/Blaizzy/mlx-vlm), MIT licence,
# Copyright (c) 2025 Prince Canuma -- full text in LICENSE.mlx-vlm beside this file.
#
#   mlx_vlm/models/gemma4/gemma4.py    sha256 c0b57f2fd9eac5b2185b4aa081606f4cb85607c1d59eaef7781931af48194b25
#     (MultimodalEmbedder)
#   mlx_vlm/models/gemma4/language.py  sha256 c15718b60a0ad6f65aefd1bd6df149469d2d09d9b0f1f534d4f750cc19367ec4
#     (RMSNormNoScale)
#
# Unchanged apart from formatting. Plain MLX ops.
"""``embed_vision``: projects vision soft tokens into the text model's embedding space."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

__all__ = ["MultimodalEmbedder", "RMSNormNoScale"]


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale (with_scale=False, scale_shift=0.0)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class MultimodalEmbedder(nn.Module):
    """Projects soft tokens from vision into language model space."""

    def __init__(self, embedding_dim: int, text_hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_projection = nn.Linear(embedding_dim, text_hidden_size, bias=False)
        self.embedding_pre_projection_norm = RMSNormNoScale(embedding_dim, eps=eps)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)
