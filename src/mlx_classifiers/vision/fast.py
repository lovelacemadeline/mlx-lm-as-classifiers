"""The ``vision_fast`` encode path (story 6.6): same weights, same arithmetic, less work.

Not vendored: a re-arrangement of :mod:`.encoder` (mlx-vlm 0.7.8's tower) for the one
case the runner has -- one image, every patch real (``VisionModel.__call__`` sets
``max_patches = num_real``, so there is never padding). The slow path stays the
reference; :class:`FastEncoder` must reproduce it, and the tests check that it does
bit for bit on a tiny random tower (CPU). What changes, per stage:

- **Preprocess**: PIL decode + bicubic resize as before (``resized_uint8``), then the
  uint8 image goes to the device as is (1/4 of the float32 bytes, no host-side
  transpose copy); ``x / 255`` and ``2 * (x - 0.5)`` are a 256-entry lookup table
  built with exactly those float32 ops, so the bf16 patch values are identical.
- **Patch positions, position embeddings, RoPE tables, pooling weights** depend only
  on ``(H, W)``: computed once per shape and reused. The position embedding is a
  two-row gather from the ``[2, 10240, D]`` table instead of a one-hot float32
  ``[1, L, 2, 10240]`` matmul (about 76 GFLOP on E2B, 115 on the 26B, and ~350 MB of
  transients per frame); a one-hot row times the table is exactly that row.
- **Attention mask**: the slow path builds an all-zero additive ``[1, 1, L, L]`` mask
  (no padding) and hands it to SDPA in every layer; here ``mask=None``.
- **No host sync**: the slow path's ``.item()`` (number of valid pooled rows, always
  ``L / 9`` here) is dropped, so tower + projector are one graph.
- **mx.compile** (``compile=True``): each layer runs as two compiled segments, traced
  once per pinned input shape -- (input norm, q/k/v projections) and (RoPE, SDPA, o_proj,
  post-attention norm, residual, MLP, residual) -- so clips, RoPE, GELU and residuals
  fuse. The float32 q/k/v RMS norms between them stay **eager**: compiled, MLX fuses
  their ``x ** 2`` differently from the eager Metal ``pow`` (about 1 ulp in float32),
  which grows to max |d| 0.08 in the soft tokens after 16 layers (story 6.6). Compiling
  the whole forward is ~4 ms faster on E2B but not bit-exact. Weights and per-shape
  tables are captured as constants: call :meth:`clear` after reloading weights.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from .encoder import _ensure_fused_sdpa, _rotate_half, one_hot
from .processor import resized_uint8

__all__ = ["FastEncoder"]


class _Shape:
    """Everything that depends only on the resized image size ``(H, W)``."""

    def __init__(self, enc: FastEncoder, height: int, width: int):
        vm = enc.vm
        p = vm.patch_size
        self.ph, self.pw = height // p, width // p
        num_real = self.ph * self.pw
        self.length = num_real // vm.pooling_kernel_size**2
        positions, padding, _ = vm._patch_positions_single(height, width, max_patches=num_real)
        assert not padding.any()  # the runner's case; the fast path assumes it
        pos = mx.array(positions[None])  # [1, L, 2] int32
        table = vm.patch_embedder.position_embedding_table
        # one_hot(pos) @ table summed over the two axes == table[0][x] + table[1][y]
        self.pos_emb = (table[0][pos[0, :, 0]] + table[1][pos[0, :, 1]])[None]
        self.rope = _rope_tables(pos, enc.head_dim, enc.rope_base)
        self.pool = _pool_weights(pos, num_real, self.length)
        mx.eval(self.pos_emb, self.rope, self.pool)
        #: Compiled ``(qkv, post)`` per layer (``compile=True``), built on first use.
        self.fns: list | None = None


def _rope_tables(positions: mx.array, head_dim: int, base: float):
    """The per-spatial-dim ``(lo, hi, cos, sin)`` of ``apply_multidimensional_rope``,
    float32 (cast to the activation dtype at use, as the slow path does)."""
    ndim = positions.shape[-1]
    cpd = 2 * (head_dim // (2 * ndim))
    out = []
    for d in range(ndim):
        freq = (2.0 / cpd) * mx.arange(0, cpd // 2).astype(mx.float32)
        inp = positions[..., d : d + 1].astype(mx.float32) / mx.power(base, freq)
        cos, sin = mx.cos(inp), mx.sin(inp)
        cos = mx.expand_dims(mx.concatenate([cos, cos], axis=-1), 2)
        sin = mx.expand_dims(mx.concatenate([sin, sin], axis=-1), 2)
        out.append((d * cpd, (d + 1) * cpd, cos, sin))
    return out


def _rope(x: mx.array, tables) -> mx.array:
    parts = []
    for lo, hi, cos, sin in tables:
        xp = x[..., lo:hi]
        parts.append(xp * cos.astype(x.dtype) + _rotate_half(xp) * sin.astype(x.dtype))
    return mx.concatenate(parts, axis=-1)


def _pool_weights(positions: mx.array, seq_len: int, length: int) -> mx.array:
    """``VisionPooler._avg_pool_by_positions``'s weights, ``[1, L, length]`` float32."""
    k = int((seq_len // length) ** 0.5)
    clamped = mx.clip(positions, 0, None)
    max_x = mx.max(clamped[..., 0], axis=-1, keepdims=True) + 1
    idx = mx.floor(clamped.astype(mx.float32) / k).astype(mx.int32)
    idx = idx[..., 0] + (max_x // k) * idx[..., 1]
    return one_hot(idx, length).astype(mx.float32) / (k**2)


class FastEncoder:
    """PNG -> ``[N, text_hidden]`` soft tokens through ``tower``'s own modules.

    ``compile``: wrap the forward in ``mx.compile`` (one trace per image shape)."""

    def __init__(self, tower: Any, compile: bool = True):
        self.tower = tower
        self.vm = tower.vision_tower
        self.compile = compile
        attn = self.vm.encoder.layers[0].self_attn
        self.head_dim = attn.head_dim
        self.rope_base = attn.rope_base_frequency
        dtype = self.vm.patch_embedder.input_proj.weight.dtype
        u8 = np.arange(256, dtype=np.uint8).astype(np.float32) * (1 / 255)  # processor's op
        self.lut = (2 * (mx.array(u8) - 0.5)).astype(dtype)  # _patchify's ops, then its cast
        self._shapes: dict[tuple[int, int], _Shape] = {}

    def clear(self) -> None:
        self._shapes.clear()

    def uint8(self, png: bytes, max_soft_tokens: int | None = None) -> np.ndarray:
        """The resized ``[H, W, 3]`` uint8 image at ``max_soft_tokens`` (default: the
        tower's); each budget is its own pinned shape (story 6.7)."""
        vc = self.tower.vision_config
        budget = max_soft_tokens or self.tower.max_soft_tokens
        return resized_uint8(png, vc.patch_size, budget, vc.pooling_kernel_size)

    def shape(self, height: int, width: int) -> _Shape:
        key = (height, width)
        if key not in self._shapes:
            self._shapes[key] = _Shape(self, height, width)
        return self._shapes[key]

    # -- stages (separately callable for the profile) --------------------------------
    def embed(self, image: mx.array, sh: _Shape) -> mx.array:
        """``[H, W, 3]`` uint8 -> ``[1, L, D]`` patch + position embeddings."""
        p = self.vm.patch_size
        x = mx.take(self.lut, image.astype(mx.int32))  # [H, W, 3]
        x = x.reshape(sh.ph, p, sh.pw, p, 3).transpose(0, 2, 1, 3, 4)
        x = x.reshape(1, sh.ph * sh.pw, p * p * 3)
        return self.vm.patch_embedder.input_proj(x) + sh.pos_emb

    def layers(self, h: mx.array, sh: _Shape) -> mx.array:
        layers = self.vm.encoder.layers
        if not self.compile:
            for layer in layers:
                h = _block(layer, h, sh.rope)
            return h
        if sh.fns is None:
            sh.fns = [
                (
                    mx.compile(lambda x, _l=layer: _qkv(_l, x)),
                    mx.compile(lambda x, q, k, v, _l=layer: _post(_l, x, q, k, v, sh.rope)),
                )
                for layer in layers
            ]
        for layer, (qkv, post) in zip(layers, sh.fns, strict=True):
            q, k, v = _norms(layer.self_attn, *qkv(h))
            h = post(h, q, k, v)
        return h

    def pool(self, h: mx.array, sh: _Shape) -> mx.array:
        """Pooler + standardize: ``[1, L, D]`` -> ``[1, N, D]``."""
        out = mx.einsum("bLl,bLd->bld", sh.pool, h).astype(h.dtype)
        out = out * self.vm.pooler.root_hidden_size
        if self.vm.config.standardize:
            out = (out - self.vm.std_bias) * self.vm.std_scale
        return out

    def project(self, h: mx.array) -> mx.array:
        return self.tower.embed_vision(h)

    def forward(self, image: mx.array, sh: _Shape) -> mx.array:
        h = self.embed(image, sh)
        h = self.layers(h, sh)
        return self.project(self.pool(h, sh))[0]

    # -- entry points ------------------------------------------------------------------
    def encode_uint8(self, image: np.ndarray | mx.array) -> mx.array:
        """``[N, text_hidden]`` (lazy) for a resized ``[H, W, 3]`` uint8 image."""
        image = mx.array(image) if not isinstance(image, mx.array) else image
        return self.forward(image, self.shape(int(image.shape[0]), int(image.shape[1])))

    def encode_png(self, png: bytes, max_soft_tokens: int | None = None) -> mx.array:
        return self.encode_uint8(self.uint8(png, max_soft_tokens))


def _qkv(layer: Any, x: mx.array):
    """Input norm and the q/k/v projections, ``[B, L, heads, head_dim]`` each."""
    attn, (B, L, _) = layer.self_attn, x.shape
    n = layer.input_layernorm(x)
    q = attn.q_proj(n).reshape(B, L, attn.num_heads, attn.head_dim)
    k = attn.k_proj(n).reshape(B, L, attn.num_kv_heads, attn.head_dim)
    v = attn.v_proj(n).reshape(B, L, attn.num_kv_heads, attn.head_dim)
    return q, k, v


def _norms(attn: Any, q: mx.array, k: mx.array, v: mx.array):
    """The float32 q/k/v RMS norms (kept out of ``mx.compile``: see module doc)."""
    return attn.q_norm(q), attn.k_norm(k), attn._v_norm(v)


def _post(layer: Any, x: mx.array, q: mx.array, k: mx.array, v: mx.array, rope) -> mx.array:
    """RoPE, SDPA (``mask=None``), o_proj, then the rest of the block."""
    attn, (B, L, _) = layer.self_attn, x.shape
    q, k = _rope(q, rope), _rope(k, rope)
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    out = _ensure_fused_sdpa(q, k, v, scale=1.0, mask=None)
    out = attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
    h = x + layer.post_attention_layernorm(out)
    ffw_out = layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h)))
    return h + ffw_out


def _block(layer: Any, x: mx.array, rope) -> mx.array:
    """``VisionTransformerBlock.__call__`` with precomputed RoPE tables and no mask."""
    return _post(layer, x, *_norms(layer.self_attn, *_qkv(layer, x)), rope)
