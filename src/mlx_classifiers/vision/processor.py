# Vendored from mlx-vlm 0.7.8 (PyPI wheel; https://github.com/Blaizzy/mlx-vlm), MIT licence,
# Copyright (c) 2025 Prince Canuma -- full text in LICENSE.mlx-vlm beside this file.
#
#   mlx_vlm/models/gemma4/processing_gemma4.py
#     sha256 44472861196712413960fc709256ba01bbdaef682af12e4119ae1bca8275c574
#     (Gemma4ImageProcessor.preprocess / aspect_ratio_preserving_resize, one PNG at a time)
#   upstream notes it is adapted from Hugging Face transformers' Gemma 4 processor
#   (Apache-2.0); the resize arithmetic lives in mlx_classifiers.images.resized_size.
#
# Changes: the transformers image utilities (fetch_images, to_numpy_array,
# infer_channel_dimension_format) are replaced by their effect on a decoded PNG:
# PIL RGB image -> uint8 HWC array. Then exactly upstream's steps: bicubic resize with
# PIL when the target differs, float32 * rescale_factor (1/255), no normalisation
# (processor_config.json: do_normalize false), channel-first.
"""PNG bytes -> ``pixel_values`` [1, 3, H, W] float32, as Gemma 4's processor builds them."""

from __future__ import annotations

import io

import numpy as np

from ..images import resized_size

__all__ = ["preprocess_png"]


def preprocess_png(
    data: bytes,
    patch_size: int = 16,
    max_soft_tokens: int = 280,
    pooling_kernel_size: int = 3,
    rescale_factor: float = 1 / 255,
) -> np.ndarray:
    """``[1, 3, H, W]`` float32 in [0, 1]; H and W multiples of ``patch_size *
    pooling_kernel_size``. Needs Pillow (the ``vision`` extra)."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    image = np.array(img)  # uint8 HWC
    height, width = image.shape[0], image.shape[1]
    max_patches = max_soft_tokens * pooling_kernel_size**2
    th, tw = resized_size(height, width, patch_size, max_patches, pooling_kernel_size)
    if (th, tw) != (height, width):
        pil = Image.fromarray(image).resize((tw, th), resample=Image.BICUBIC)
        image = np.array(pil)
    image = image.astype(np.float32) * rescale_factor
    return np.transpose(image, (2, 0, 1))[None]
