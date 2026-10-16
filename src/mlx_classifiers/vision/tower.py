"""Load the vision weights mlx-lm drops, and encode one PNG to soft tokens (story 6.2).

mlx-lm 0.31.3's ``gemma4.Model.sanitize`` skips every ``vision_tower``,
``embed_vision``, ``multi_modal_projector`` and audio key, so ``mlx_lm.load``
never sees them. :func:`load_vision_tower` reads exactly those tensors from the
same snapshot's safetensors shards (``mx.load`` is lazy: only the selected
arrays are ever read), applies mlx-vlm's sanitize rules for them (drop
``rotary_emb``; keep the clip bounds only when ``use_clipped_linears``), and
quantizes a module exactly when the checkpoint holds ``<path>.scales`` for it,
at the checkpoint's (per-path or global) group size and bits. On the E2B QAT
checkpoint the tower is bf16 and ``embed_vision.embedding_projection`` is
4-bit affine, group 64.

No mlx-vlm import anywhere: the modules are the vendored copies beside this
file.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from .encoder import VisionConfig, VisionModel
from .processor import preprocess_png
from .projector import MultimodalEmbedder

__all__ = ["VISION_PREFIXES", "VisionTower", "load_vision_tower", "vision_weights"]

#: Checkpoint key prefixes of the image path (mlx-lm's gemma4 sanitize drops them).
VISION_PREFIXES = ("vision_tower.", "embed_vision.")
_CLIP_KEYS = ("input_max", "input_min", "output_max", "output_min")


class VisionTower(nn.Module):
    """``vision_tower`` + ``embed_vision`` under their checkpoint names, plus the
    processor numbers. :meth:`encode_png` returns ``[N, text_hidden]`` soft tokens
    in the text model's embedding space (bf16), memoised by (PNG sha256, budget).

    ``max_soft_tokens`` is the default soft-token budget (the checkpoint's 280); a
    call may pass another (story 6.7: 70 / 140 / 280). The budget sets the resize, so
    the patch count (``<= budget * 9``), the tower's sequence length and the output
    count (``patches / 9``) all follow it."""

    def __init__(
        self,
        config: dict[str, Any],
        text_hidden_size: int,
        cache_size: int = 16,
        fast: bool = False,
        compile: bool = True,
    ):
        super().__init__()
        vc = VisionConfig.from_dict(config["vision_config"])
        self.vision_config = vc
        self.vision_tower = VisionModel(vc)
        self.embed_vision = MultimodalEmbedder(
            embedding_dim=vc.hidden_size, text_hidden_size=text_hidden_size, eps=vc.rms_norm_eps
        )
        self.max_soft_tokens = int(config.get("vision_soft_tokens_per_image", 280))
        # Plain attributes: nn.Module would register a dict as a parameter container.
        object.__setattr__(self, "_features", OrderedDict())
        self._cache_size = cache_size
        self._lock = threading.Lock()
        #: Encodes actually run (cache misses), for tests and the bench.
        self.encodes = 0
        #: Story 6.6: encode through :class:`~.fast.FastEncoder` (same numerics, less
        #: work); False keeps the vendored path, which stays the reference.
        self.fast = fast
        self._compile = compile
        object.__setattr__(self, "_fast_encoder", None)

    def fast_encoder(self):
        """The :class:`~.fast.FastEncoder` over these weights (built on first use)."""
        if self._fast_encoder is None:
            from .fast import FastEncoder

            object.__setattr__(self, "_fast_encoder", FastEncoder(self, compile=self._compile))
        return self._fast_encoder

    def pixel_values(self, png: bytes, max_soft_tokens: int | None = None) -> mx.array:
        vc = self.vision_config
        budget = max_soft_tokens or self.max_soft_tokens
        pixels = preprocess_png(png, vc.patch_size, budget, vc.pooling_kernel_size)
        return mx.array(pixels)

    def encode_pixels(self, pixel_values: mx.array) -> mx.array:
        """``[1, N, text_hidden]`` for ``[1, 3, H, W]`` pixels (mlx-vlm ``encode_image``)."""
        return self.embed_vision(self.vision_tower(pixel_values))

    def encode_png(
        self, png: bytes, sha256: str | None = None, max_soft_tokens: int | None = None
    ) -> mx.array:
        budget = max_soft_tokens or self.max_soft_tokens
        key = (sha256 or hashlib.sha256(png).hexdigest(), budget)
        with self._lock:
            hit = self._features.get(key)
            if hit is not None:
                self._features.move_to_end(key)
                return hit
        if self.fast:
            feats = self.fast_encoder().encode_png(png, budget)
        else:
            feats = self.encode_pixels(self.pixel_values(png, budget))[0]
        mx.eval(feats)
        with self._lock:
            self.encodes += 1
            self._features[key] = feats
            while len(self._features) > self._cache_size:
                self._features.popitem(last=False)
        return feats

    def clear(self) -> None:
        with self._lock:
            self._features.clear()


def _snapshot_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = {
            f for k, f in weight_map.items() if k.removeprefix("model.").startswith(VISION_PREFIXES)
        }
        return [model_dir / n for n in sorted(names)]
    return sorted(model_dir.glob("*.safetensors"))


def vision_weights(model_dir: str | Path, use_clipped_linears: bool) -> dict[str, mx.array]:
    """The image-path tensors of a snapshot, under ``vision_tower.*`` / ``embed_vision.*``."""
    model_dir = Path(model_dir)
    out: dict[str, mx.array] = {}
    for path in _snapshot_files(model_dir):
        for k, v in mx.load(str(path)).items():
            k = k.removeprefix("model.")
            if not k.startswith(VISION_PREFIXES) or "rotary_emb" in k:
                continue
            if any(s in k for s in _CLIP_KEYS) and not use_clipped_linears:
                continue
            out[k] = v
    if not out:
        raise ValueError(f"{model_dir} holds no vision_tower / embed_vision weights")
    return out


def _quant_params(quant: dict[str, Any], path: str) -> dict[str, Any] | None:
    own = quant.get(path)
    if isinstance(own, dict):
        return {k: own[k] for k in ("group_size", "bits", "mode") if k in own}
    return None


def load_vision_tower(
    model_dir: str | Path,
    text_hidden_size: int | None = None,
    cache_size: int = 16,
    fast: bool = False,
    compile: bool = True,
) -> VisionTower:
    """Build the vendored tower for a Gemma 4 snapshot directory and load its weights
    (strict: every parameter must be in the checkpoint and vice versa). ``fast`` /
    ``compile``: the story 6.6 encode path (:class:`VisionTower` ``fast``)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if not config.get("vision_config"):
        raise ValueError(f"{model_dir} has no vision_config")
    if text_hidden_size is None:
        text_hidden_size = int(config["text_config"]["hidden_size"])
    tower = VisionTower(config, text_hidden_size, cache_size, fast=fast, compile=compile)
    weights = vision_weights(model_dir, tower.vision_config.use_clipped_linears)
    quant = config.get("quantization") or config.get("quantization_config")
    if quant:

        def predicate(path: str, module: nn.Module) -> bool | dict[str, Any]:
            if not hasattr(module, "to_quantized") or f"{path}.scales" not in weights:
                return False
            return _quant_params(quant, path) or True

        nn.quantize(
            tower,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
            class_predicate=predicate,
        )
    tower.load_weights(list(weights.items()), strict=True)
    mx.eval(tower.parameters())
    tower.eval()
    return tower
