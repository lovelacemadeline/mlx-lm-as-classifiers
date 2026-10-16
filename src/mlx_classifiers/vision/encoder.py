# Vendored from mlx-vlm 0.7.8 (PyPI wheel; https://github.com/Blaizzy/mlx-vlm), MIT licence,
# Copyright (c) 2025 Prince Canuma -- full text in LICENSE.mlx-vlm beside this file.
#
#   mlx_vlm/models/gemma4/vision.py   sha256 f3e61e075c568e2f2188629e3ad83131e523a040738005fa6b5fd4e51ffd0293
#   mlx_vlm/models/gemma4/config.py   sha256 f4d1194e9f8329a37a686c273e06d13a79f4e69efd0b5474197bdf9cc2c0b954
#     (VisionConfig only)
#   mlx_vlm/models/base.py            sha256 a7cfa38413bea057946c5d56c9edd34dbcceb33f6613529f5201ca2ad5b41120
#     (ensure_fused_sdpa only)
#   mlx_vlm/models/rope_utils.py      sha256 6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b
#     (rotate_half only)
#
# Changes from upstream (story 6.2), numerics untouched:
# - the list-of-images, pre-patchified (pixel_position_ids) and video branches of
#   VisionModel.__call__ are dropped: one channel-first image batch [B, C, H, W] only;
# - ensure_fused_sdpa and rotate_half are inlined; VisionConfig is a plain dataclass with
#   ``from_dict`` (upstream's BaseModelConfig); the unused ``sanitize`` staticmethod is dropped.
# Every op is plain MLX (mx.fast.scaled_dot_product_attention / rms_norm, einsum, matmul).
"""Gemma 4 vision tower (SigLIP-style ViT with 2D RoPE and position-pooled output)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

__all__ = ["VisionConfig", "VisionModel"]


@dataclass
class VisionConfig:
    model_type: str = "gemma4_vision"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    global_head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_bidirectional_attention: str = "vision"
    layer_types: list[str] | None = None
    rope_parameters: dict | None = None
    default_output_length: int = 280
    patch_size: int = 16
    position_embedding_size: int = 10240
    pooling_kernel_size: int = 3
    use_clipped_linears: bool = False
    standardize: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 100.0, "rope_type": "default"}

    @classmethod
    def from_dict(cls, params: dict[str, Any] | None) -> VisionConfig:
        if not params:
            return cls()
        names = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in names})


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _ensure_fused_sdpa(q, k, v, scale, mask=None):
    fused_dims = (64, 80, 128)  # supported by MLX's fused SDPA kernel
    d = q.shape[-1]
    target = next((t for t in fused_dims if d <= t), d)
    if target != d:
        pad = [(0, 0)] * (q.ndim - 1) + [(0, target - d)]
        q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)[..., :d]


class ClippableLinear(nn.Module):
    """Linear layer with optional input/output clamping (Gemma4ClippableLinear).

    Clip bounds are scalar buffers in the checkpoint; +-inf until loaded."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, use_clipping: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_clipping = use_clipping
        if use_clipping:
            self.input_min = mx.array(float("-inf"))
            self.input_max = mx.array(float("inf"))
            self.output_min = mx.array(float("-inf"))
            self.output_max = mx.array(float("inf"))

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_clipping:
            x = mx.clip(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipping:
            x = mx.clip(x, self.output_min, self.output_max)
        return x


def one_hot(indices: mx.array, num_classes: int) -> mx.array:
    return (mx.expand_dims(indices, -1) == mx.arange(num_classes)).astype(mx.float32)


class VisionRMSNorm(nn.Module):
    """RMS norm with learned scale, computed in float32 (Gemma4RMSNorm with_scale=True)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        var = mx.mean(x_float**2, axis=-1, keepdims=True)
        normed = x_float * mx.rsqrt(var + self.eps)
        result = normed * self.weight.astype(mx.float32)
        return result.astype(x.dtype)


class VisionRMSNormNoScale(nn.Module):
    """Parameter-free RMS norm in float32 (Gemma4RMSNorm with_scale=False)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        var = mx.mean(x_float**2, axis=-1, keepdims=True)
        return (x_float * mx.rsqrt(var + self.eps)).astype(x.dtype)


class RMSNorm(nn.Module):
    """Standard Gemma 4 RMSNorm: weight applied directly."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def apply_multidimensional_rope(inputs, positions, base_frequency=100.0):
    """2D RoPE: the head dim is split per spatial dimension and each part rotated
    independently. ``inputs`` [B, L, N, H], ``positions`` [B, L, 2] (or [B, L])."""
    head_dim = inputs.shape[-1]

    if positions.ndim == 2:
        half = head_dim // 2
        freq_exponents = (2.0 / head_dim) * mx.arange(0, half).astype(mx.float32)
        timescale = mx.power(base_frequency, freq_exponents)
        sinusoid_inp = positions[..., None].astype(mx.float32) / timescale
        cos_val = mx.cos(sinusoid_inp)
        sin_val = mx.sin(sinusoid_inp)
        cos_val = mx.concatenate([cos_val, cos_val], axis=-1).astype(inputs.dtype)
        sin_val = mx.concatenate([sin_val, sin_val], axis=-1).astype(inputs.dtype)
        cos_val = mx.expand_dims(cos_val, axis=2)
        sin_val = mx.expand_dims(sin_val, axis=2)
        return inputs * cos_val + _rotate_half(inputs) * sin_val

    ndim = positions.shape[-1]
    channels_per_dim = 2 * (head_dim // (2 * ndim))
    half_per_dim = channels_per_dim // 2

    result_parts = []
    for d in range(ndim):
        x_part = inputs[..., d * channels_per_dim : (d + 1) * channels_per_dim]
        freq_exponents = (2.0 / channels_per_dim) * mx.arange(0, half_per_dim).astype(mx.float32)
        timescale = mx.power(base_frequency, freq_exponents)
        sinusoid_inp = positions[..., d : d + 1].astype(mx.float32) / timescale
        cos_d = mx.cos(sinusoid_inp)
        sin_d = mx.sin(sinusoid_inp)
        cos_d = mx.concatenate([cos_d, cos_d], axis=-1).astype(inputs.dtype)
        sin_d = mx.concatenate([sin_d, sin_d], axis=-1).astype(inputs.dtype)
        cos_d = mx.expand_dims(cos_d, axis=2)
        sin_d = mx.expand_dims(sin_d, axis=2)
        y_part = x_part * cos_d + _rotate_half(x_part) * sin_d
        result_parts.append(y_part)

    return mx.concatenate(result_parts, axis=-1)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rope_base_frequency = config.rope_parameters["rope_theta"]

        clip = getattr(config, "use_clipped_linears", False)
        h, d = self.hidden_size, self.head_dim
        self.q_proj = ClippableLinear(h, self.num_heads * d, bias=False, use_clipping=clip)
        self.k_proj = ClippableLinear(h, self.num_kv_heads * d, bias=False, use_clipping=clip)
        self.v_proj = ClippableLinear(h, self.num_kv_heads * d, bias=False, use_clipping=clip)
        self.o_proj = ClippableLinear(self.num_heads * d, h, bias=False, use_clipping=clip)

        self.q_norm = VisionRMSNorm(self.head_dim)
        self.k_norm = VisionRMSNorm(self.head_dim)
        self._v_norm = VisionRMSNormNoScale()

    def __call__(self, x: mx.array, positions: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self._v_norm(v)

        q = apply_multidimensional_rope(q, positions, self.rope_base_frequency)
        k = apply_multidimensional_rope(k, positions, self.rope_base_frequency)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn_output = _ensure_fused_sdpa(q, k, v, scale=1.0, mask=mask)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_output)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        clip = getattr(config, "use_clipped_linears", False)
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = ClippableLinear(h, i, bias=False, use_clipping=clip)
        self.up_proj = ClippableLinear(h, i, bias=False, use_clipping=clip)
        self.down_proj = ClippableLinear(i, h, bias=False, use_clipping=clip)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class VisionTransformerBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def __call__(self, x: mx.array, positions: mx.array, mask: mx.array | None = None) -> mx.array:
        normed = self.input_layernorm(x)
        attn_out = self.self_attn(normed, positions, mask)
        attn_out = self.post_attention_layernorm(attn_out)
        h = x + attn_out

        normed_h = self.pre_feedforward_layernorm(h)
        ffw_out = self.mlp(normed_h)
        ffw_out = self.post_feedforward_layernorm(ffw_out)
        return h + ffw_out


class VisionPatchEmbedder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size, bias=False)
        self.position_embedding_table = mx.ones((2, self.position_embedding_size, self.hidden_size))

    def _position_embeddings(self, patch_positions: mx.array, padding_positions: mx.array):
        oh = one_hot(patch_positions, self.position_embedding_size)
        # [B, num_patches, 2, pos_size] -> [B, 2, num_patches, pos_size]
        oh = oh.transpose(0, 2, 1, 3).astype(self.position_embedding_table.dtype)
        position_embeddings = oh @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(axis=1)
        return mx.where(mx.expand_dims(padding_positions, -1), 0.0, position_embeddings)

    def _patchify(self, pixel_values: mx.array) -> mx.array:
        # [B, C, H, W] -> [B, pH, pW, p, p, C] -> [B, pH*pW, p*p*C]
        B, C, H, W = pixel_values.shape
        p = self.patch_size
        pH, pW = H // p, W // p
        patches = pixel_values.reshape(B, C, pH, p, pW, p)
        patches = patches.transpose(0, 2, 4, 3, 5, 1)
        patches = patches.reshape(B, pH * pW, C * p * p)
        patches = 2 * (patches - 0.5)
        return self.input_proj(patches.astype(self.input_proj.weight.dtype))

    def __call__(self, pixel_values, patch_positions, padding_positions) -> mx.array:
        hidden_states = self._patchify(pixel_values)
        return hidden_states + self._position_embeddings(patch_positions, padding_positions)


class VisionPooler(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.default_output_length = config.default_output_length
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(self, x, patch_positions, length):
        input_seq_len = x.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2

        clamped = mx.clip(patch_positions, 0, None)
        max_x = mx.max(clamped[..., 0], axis=-1, keepdims=True) + 1
        kernel_idxs = mx.floor(clamped.astype(mx.float32) / k).astype(mx.int32)
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = one_hot(kernel_idxs, length).astype(mx.float32) / k_squared
        output = mx.einsum("bLl,bLd->bld", weights, x).astype(x.dtype)
        mask = mx.logical_not(mx.all(weights == 0, axis=1))
        return output, mask

    def __call__(self, hidden_states, patch_positions, padding_positions, output_length=None):
        hidden_states = mx.where(mx.expand_dims(padding_positions, -1), 0.0, hidden_states)
        length = output_length or self.default_output_length
        if hidden_states.shape[1] == length:
            mask = padding_positions
        else:
            hidden_states, mask = self._avg_pool_by_positions(
                hidden_states, patch_positions, length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, mask


class VisionTransformerModel(nn.Module):
    """The transformer layers (``vision_tower.encoder`` in the checkpoint)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [VisionTransformerBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states: mx.array, positions: mx.array, mask: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, mask)
        return hidden_states


class VisionModel(nn.Module):
    """``vision_tower.*``: patch embedder, encoder layers, position pooler.

    ``__call__(pixel_values [B, 3, H, W])`` returns ``[1, B * N, hidden]`` soft
    tokens, ``N = (H/16) * (W/16) / 9`` per image."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.patch_size = config.patch_size
        self.pooling_kernel_size = config.pooling_kernel_size
        self.default_output_length = config.default_output_length
        self.max_patches = self.default_output_length * self.pooling_kernel_size**2

        self.patch_embedder = VisionPatchEmbedder(config)
        self.encoder = VisionTransformerModel(config)
        self.pooler = VisionPooler(config)

        if config.standardize:
            self.std_bias = mx.zeros((config.hidden_size,))
            self.std_scale = mx.ones((config.hidden_size,))

    def _patch_positions_single(self, H, W, max_patches=None):
        """Patch positions and padding mask for one image."""
        if max_patches is None:
            max_patches = self.max_patches
        pH = H // self.patch_size
        pW = W // self.patch_size
        num_patches = pH * pW

        gx, gy = np.meshgrid(np.arange(pW), np.arange(pH), indexing="xy")
        real_positions = np.stack([gx.flatten(), gy.flatten()], axis=-1)

        num_padding = max_patches - num_patches
        if num_padding > 0:
            pad_positions = np.full((num_padding, 2), -1, dtype=np.int64)
            patch_positions = np.concatenate([real_positions, pad_positions], axis=0)
        else:
            patch_positions = real_positions[:max_patches]

        padding_mask = np.zeros(max_patches, dtype=bool)
        if num_padding > 0:
            padding_mask[num_patches:] = True

        return patch_positions.astype(np.int32), padding_mask, num_patches

    def __call__(self, pixel_values: mx.array) -> mx.array:
        if not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)

        pool_sq = self.pooling_kernel_size**2
        B, _, H, W = pixel_values.shape
        num_real = (H // self.patch_size) * (W // self.patch_size)
        output_length = num_real // pool_sq
        positions, padding_mask, _ = self._patch_positions_single(H, W, max_patches=num_real)
        patch_positions = mx.array(np.tile(positions[None], (B, 1, 1)))
        padding_positions = mx.array(np.tile(padding_mask[None], (B, 1)))

        inputs_embeds = self.patch_embedder(pixel_values, patch_positions, padding_positions)

        # Bidirectional attention mask [B, 1, L, L]; -1e4 rather than -inf (upstream note:
        # all-masked padded rows give NaN gradients).
        valid_mask = ~padding_positions
        attn_mask = mx.expand_dims(valid_mask, 1) * mx.expand_dims(valid_mask, 2)
        mask_fill = mx.array(-1e4, dtype=inputs_embeds.dtype)
        attn_mask = mx.where(attn_mask, mx.array(0.0, dtype=inputs_embeds.dtype), mask_fill)
        attn_mask = mx.expand_dims(attn_mask, 1)

        hidden_states = self.encoder(inputs_embeds, patch_positions, attn_mask)

        pooled, pool_mask = self.pooler(
            hidden_states, patch_positions, padding_positions, output_length=output_length
        )

        valid_mask = pool_mask if pool_mask.shape[1] == output_length else ~pool_mask

        all_real = []
        for i in range(B):
            n_valid = int(valid_mask[i].astype(mx.int32).sum().item())
            all_real.append(pooled[i, :n_valid])

        hidden_states = mx.concatenate(all_real, axis=0)[None]

        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states
