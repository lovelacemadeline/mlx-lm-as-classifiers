"""Image blocks in the state: decoding, soft-token counts, placeholder runs (story 6.1).

Weightless and dependency-free (no PIL, no mlx): everything prompt compilation
and the delta cache need to know about an image is its PNG bytes, their
sha256, and the number of soft tokens the model's processor will give it.

Gemma 4's convention (verified against the E2B checkpoint's config.json,
processor_config.json and the mlx-vlm 0.7.8 processor, story 6.1): the chat
template marks an image with ``<|image|>``; the processor expands it to
``<|image>`` (boi, id 255999) + ``N`` x ``<|image|>`` (image token, id 258880)
+ ``<image|>`` (eoi, id 258882). ``N`` is not fixed: the image is resized,
aspect ratio kept, to the largest size whose sides are multiples of
``patch_size * pooling_kernel_size`` (48 px) and that holds at most
``max_soft_tokens * pooling_kernel_size**2`` patches (2520), and ``N`` is its
patch count over ``pooling_kernel_size**2``. A 160 x 192 gridworld frame
becomes 720 x 864 and 270 soft tokens; a square image 768 x 768 and 256.

:func:`keyed_tokens` is the delta cache's view of a prompt: every placeholder
id inside an image run is replaced by a negative id derived from that image's
sha256, so two prompts that differ only in their image diverge at the image's
first soft token, and the same image at the same place still shares the
prefix.

**Soft-token budget** (story 6.7). ``max_soft_tokens`` is a budget, not a count:
the processor resizes to at most ``budget * 9`` patches, and the tower pools
those 3 x 3 to ``N <= budget`` soft tokens. Gemma 4 supports 70, 140, 280 (the
checkpoint default), 560 and 1120; this library supports :data:`SOFT_TOKEN_BUDGETS`
(70, 140, 280). A 160 x 192 gridworld frame is 720 x 864 = 2430 patches, 270
tokens at 280; 480 x 576 = 1080 patches, 120 tokens at 140; 336 x 432 = 567
patches, 63 tokens at 70. Tower cost follows the patch count (attention is
quadratic in it), so 140 is 2.25x fewer patches and 70 is 4.3x fewer. A slot
records the budget it was compiled at, and the delta-cache key includes it, so the
same PNG at two budgets never shares KV.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .errors import PromptError

__all__ = [
    "DEFAULT_SOFT_TOKENS",
    "IMAGE_BLOCK_TYPE",
    "SOFT_TOKEN_BUDGETS",
    "ImageSlot",
    "ImageTokens",
    "decode_png_base64",
    "find_image_runs",
    "image_key_id",
    "image_tokens_from_config",
    "keyed_tokens",
    "png_size",
    "resized_size",
    "soft_token_count",
    "write_png",
]

IMAGE_BLOCK_TYPE = "image/png"
#: Gemma 4's processor default (``max_soft_tokens`` / ``vision_soft_tokens_per_image``).
DEFAULT_SOFT_TOKENS = 280
#: The soft-token budgets this library supports per image (story 6.7).
SOFT_TOKEN_BUDGETS = (70, 140, 280)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int]:
    """``(width, height)`` from a PNG's IHDR chunk; :class:`PromptError` if not a PNG."""
    if len(data) < 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise PromptError("image block is not a PNG (bad signature or missing IHDR)")
    width, height = struct.unpack(">II", data[16:24])
    if width == 0 or height == 0:
        raise PromptError("image block is an empty PNG")
    return width, height


def decode_png_base64(text: str) -> bytes:
    """The PNG bytes of an image block's ``base64`` field (validated as a PNG)."""
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PromptError(f"image block base64 does not decode: {exc}") from None
    png_size(data)
    return data


def resized_size(
    height: int, width: int, patch_size: int = 16, max_patches: int = 2520, pooling: int = 3
) -> tuple[int, int]:
    """``(height, width)`` Gemma 4's image processor resizes to.

    Vendored arithmetic: ``get_aspect_ratio_preserving_size`` from mlx-vlm 0.7.8
    ``mlx_vlm/models/gemma4/processing_gemma4.py`` (MIT, Prince Canuma; file
    sha256 44472861196712413960fc709256ba01bbdaef682af12e4119ae1bca8275c574),
    itself adapted from Hugging Face transformers' Gemma 4 processor."""
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / (height * width))
    side_mult = pooling * patch_size
    target_height = math.floor(factor * height / side_mult) * side_mult
    target_width = math.floor(factor * width / side_mult) * side_mult
    if target_height == 0 and target_width == 0:
        raise PromptError("image resizes to 0 x 0")
    max_side_length = (max_patches // pooling**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(math.floor(width / height) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(math.floor(height / width) * side_mult, max_side_length)
    return target_height, target_width


def soft_token_count(
    height: int, width: int, patch_size: int = 16, max_soft_tokens: int = 280, pooling: int = 3
) -> int:
    """Soft tokens Gemma 4 gives an image of this size (see the module doc)."""
    h, w = resized_size(height, width, patch_size, max_soft_tokens * pooling**2, pooling)
    return (h // patch_size) * (w // patch_size) // (pooling**2)


@dataclass(frozen=True)
class ImageTokens:
    """A model's image placeholder convention: the token strings of the run
    ``boi + image * N + eoi``, the image token's id, and the processor numbers
    that fix ``N``. Runners with a vision path expose one as ``image_tokens``."""

    boi: str
    image: str
    eoi: str
    image_token_id: int
    patch_size: int = 16
    pooling_kernel_size: int = 3
    max_soft_tokens: int = 280

    def soft_tokens(self, png: bytes) -> int:
        width, height = png_size(png)
        return soft_token_count(
            height, width, self.patch_size, self.max_soft_tokens, self.pooling_kernel_size
        )

    def placeholder(self, png: bytes) -> str:
        """The text the template's image marker expands to for this image."""
        return self.boi + self.image * self.soft_tokens(png) + self.eoi


def image_tokens_from_config(model_dir: str | Path) -> ImageTokens | None:
    """The :class:`ImageTokens` of a Hugging Face snapshot directory, from its
    ``config.json`` (``image_token_id``, ``vision_config``), ``tokenizer_config.json``
    (``boi_token``, ``image_token``, ``eoi_token``) and ``processor_config.json``
    (``patch_size``, ``pooling_kernel_size``, ``max_soft_tokens``). None when the
    checkpoint has no vision tower. Reads JSON only, no weights."""
    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text())
    if not config.get("vision_config") or config.get("image_token_id") is None:
        return None
    tok = json.loads((d / "tokenizer_config.json").read_text())
    proc_path = d / "processor_config.json"
    proc = (
        json.loads(proc_path.read_text()).get("image_processor", {}) if proc_path.exists() else {}
    )
    vision = config["vision_config"]
    return ImageTokens(
        boi=tok["boi_token"],
        image=tok["image_token"],
        eoi=tok["eoi_token"],
        image_token_id=int(config["image_token_id"]),
        patch_size=int(proc.get("patch_size", vision.get("patch_size", 16))),
        pooling_kernel_size=int(
            proc.get("pooling_kernel_size", vision.get("pooling_kernel_size", 3))
        ),
        max_soft_tokens=int(
            proc.get("max_soft_tokens", config.get("vision_soft_tokens_per_image", 280))
        ),
    )


@dataclass(frozen=True)
class ImageSlot:
    """One image in a compiled prompt: its soft-token run is
    ``tokens[start : start + length]`` (the image-token ids between boi and eoi);
    ``sha256`` is the hash of the PNG bytes, which ride along in ``data``.
    ``max_soft_tokens``: the budget the run was compiled at (story 6.7); None means
    the runner's own."""

    start: int
    length: int
    sha256: str
    data: bytes = field(default=b"", repr=False, compare=False)
    max_soft_tokens: int | None = None

    @property
    def end(self) -> int:
        return self.start + self.length

    @classmethod
    def from_png(
        cls, start: int, length: int, data: bytes, max_soft_tokens: int | None = None
    ) -> ImageSlot:
        return cls(start, length, hashlib.sha256(data).hexdigest(), data, max_soft_tokens)

    def describe(self) -> dict[str, object]:
        """The fixture/wire record: ``{start, length, sha256}``, plus ``max_soft_tokens``
        when the budget is not the default 280 (so default fixtures are unchanged)."""
        out: dict[str, object] = {"start": self.start, "length": self.length, "sha256": self.sha256}
        if self.max_soft_tokens is not None and self.max_soft_tokens != DEFAULT_SOFT_TOKENS:
            out["max_soft_tokens"] = self.max_soft_tokens
        return out


def find_image_runs(tokens: Sequence[int], image_token_id: int) -> list[tuple[int, int]]:
    """``(start, length)`` of each maximal run of ``image_token_id`` in ``tokens``."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, t in enumerate(tokens):
        if t == image_token_id:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(tokens) - start))
    return runs


def image_key_id(sha256: str, max_soft_tokens: int | None = None) -> int:
    """The negative stand-in id for an image's soft tokens in a delta-cache key.
    With a budget (story 6.7) the id also depends on it: the same PNG at 140 and 280
    is different soft tokens and must not share a cached prefix."""
    if max_soft_tokens is None:
        return -1 - int(sha256[:15], 16)
    digest = hashlib.sha256(f"{sha256}:{max_soft_tokens}".encode()).hexdigest()
    return -1 - int(digest[:15], 16)


def keyed_tokens(tokens: Sequence[int], images: Sequence[ImageSlot] = ()) -> tuple[int, ...]:
    """``tokens`` with each image run's ids replaced by :func:`image_key_id`."""
    out = tuple(tokens)
    if not images:
        return out
    key = list(out)
    for slot in images:
        if slot.end > len(key):
            raise ValueError(f"image run [{slot.start}, {slot.end}) beyond {len(key)} tokens")
        key[slot.start : slot.end] = [image_key_id(slot.sha256, slot.max_soft_tokens)] * slot.length
    return tuple(key)


def write_png(width: int, height: int, rgb: Sequence[Sequence[tuple[int, int, int]]]) -> bytes:
    """A minimal 8-bit RGB PNG (no filter, deterministic zlib level 9) from rows of
    ``(r, g, b)`` pixels. For tests and synthetic fixtures; no PIL needed."""
    if len(rgb) != height or any(len(row) != width for row in rgb):
        raise ValueError("pixel rows do not match width x height")
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in rgb)

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
