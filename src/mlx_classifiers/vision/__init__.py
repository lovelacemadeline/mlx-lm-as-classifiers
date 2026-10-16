"""The Gemma 4 image path, vendored (Epic 6, story 6.2): no mlx-vlm at runtime.

- :mod:`.encoder` -- the vision tower (``vision_tower.*``), from mlx-vlm 0.7.8;
- :mod:`.projector` -- ``embed_vision``, the projection into text embedding space;
- :mod:`.processor` -- PNG -> pixel values (aspect-preserving bicubic resize, /255);
- :mod:`.merge` -- soft tokens merged at the placeholder positions, per-layer token
  ids, the bidirectional image-block mask, and the decoder layer loop on embeddings;
- :mod:`.tower` -- loading the vision weights mlx-lm's sanitize drops.

Each vendored file carries a provenance header (source, version, file sha256,
licence); mlx-vlm's MIT licence text is ``LICENSE.mlx-vlm`` here. Importing this
package needs ``mlx`` and, to decode PNGs, Pillow (the ``vision`` extra).
Nothing imports it on a text-only path: ``MLXRunner`` loads it on the first image.
"""

from .tower import VisionTower, load_vision_tower

__all__ = ["VisionTower", "load_vision_tower"]
