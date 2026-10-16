---
type: story
status: completed
title: Vendor the Gemma 4 vision encoder, projector and merge with provenance
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/vision/encoder.py::VisionModel"
  - symbol: "src/mlx_classifiers/vision/encoder.py::VisionConfig"
  - symbol: "src/mlx_classifiers/vision/projector.py::MultimodalEmbedder"
  - symbol: "src/mlx_classifiers/vision/processor.py::preprocess_png"
  - symbol: "src/mlx_classifiers/vision/merge.py::merged_embeddings"
  - symbol: "src/mlx_classifiers/vision/merge.py::per_layer_token_ids"
  - symbol: "src/mlx_classifiers/vision/merge.py::forward_embeddings"
  - symbol: "src/mlx_classifiers/vision/tower.py::VisionTower"
  - symbol: "src/mlx_classifiers/vision/tower.py::load_vision_tower"
  - symbol: "src/mlx_classifiers/vision/tower.py::vision_weights"
  - test: "tests/test_vision_vendored.py::test_vendored_files_carry_provenance_headers"
  - test: "tests/test_vision_vendored.py::test_mlx_vlm_licence_text_ships_beside_the_code"
  - test: "tests/test_vision_vendored.py::test_no_module_imports_mlx_vlm"
  - test: "tests/test_vision_vendored.py::test_vision_path_runs_with_mlx_vlm_unimportable"
  - test: "tests/test_vision_vendored.py::test_text_only_runner_import_loads_nothing_vision_related"
  - test: "tests/test_vision_mlx.py::test_tower_loads_every_vision_weight_and_quantizes_only_the_projector"
  - test: "tests/test_vision_mlx.py::test_encode_a_frame_gives_270_soft_tokens_deterministically"
---

# Story 6.2 — Vendor the Gemma 4 vision encoder, projector and merge with provenance

Nobody had vendored Gemma vision on MLX before (research doc §B); mlx-lm's
`gemma4.py` sanitize drops `vision_tower`, `embed_vision`,
`multi_modal_projector` and audio keys. This story copies the minimal image
path from **mlx-vlm 0.7.8** (PyPI wheel, `https://github.com/Blaizzy/mlx-vlm`,
read in a throwaway venv only) into `src/mlx_classifiers/vision/`. mlx-vlm is
never imported at runtime; the dependency fallback was not needed.

## What was vendored, from where

| our file | upstream file (mlx-vlm 0.7.8) | upstream sha256 | what |
|---|---|---|---|
| `vision/encoder.py` | `models/gemma4/vision.py` (+ `VisionConfig` from `config.py`, `ensure_fused_sdpa` from `base.py`, `rotate_half` from `rope_utils.py`) | `f3e61e07…` (+ `f4d1194e…`, `a7cfa384…`, `6944f69a…`) | the tower: patch embedder with 2D position table, 16 (E2B) / 27 (26B) bidirectional ViT blocks with 2D RoPE and clippable linears, position-pooled 3x3 output, optional standardize (26B) |
| `vision/projector.py` | `models/gemma4/gemma4.py` (`MultimodalEmbedder`), `language.py` (`RMSNormNoScale`) | `c0b57f2f…`, `c15718b6…` | `embed_vision`: scale-free RMS norm, then the projection into the text hidden size |
| `vision/processor.py` | `models/gemma4/processing_gemma4.py` (`Gemma4ImageProcessor`) | `44472861…` | PNG -> RGB uint8 -> bicubic resize (PIL) to the aspect-preserving 48-px-multiple size -> x 1/255 -> channel-first; the size arithmetic is `images.resized_size` (story 6.1) |
| `vision/merge.py` | `gemma4.py` (`get_input_embeddings`), `language.py` (block mask overlay); the layer loop from mlx-lm 0.31.3 `gemma4_text.py` (`77f46bc3…`, MIT, Apple) | as left | merged embeddings, per-layer ids (image positions -> pad id 0), the bidirectional image-block mask, the decoder loop on embeddings (used by 6.3) |

Full hashes are in each file's header. Changes are structural only: the
list-of-images, pre-patchified and video branches of `VisionModel.__call__`
are dropped (one `[B, 3, H, W]` batch), helpers inlined, the transformers
image utilities replaced by their effect on a decoded PNG. Every op is plain
MLX (`mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm`, `einsum`,
matmul, `nn.QuantizedLinear`); no custom kernels.

**Weight loading** (`vision/tower.py`): `load_vision_tower(snapshot_dir)`
wraps rather than patches mlx-lm. It reads only the `vision_tower.*` /
`embed_vision.*` tensors from the shards the index names (`mx.load` is lazy),
applies mlx-vlm's sanitize rules for them (drop `rotary_emb`; keep the clip
bounds only when `use_clipped_linears`: true on E2B, false on the 26B),
quantizes a module exactly when the checkpoint has its `.scales` (E2B and 26B:
the tower is bf16, `embed_vision.embedding_projection` is 4-bit affine group
64), and loads strictly. Load takes 0.05 s on E2B, 0.35 s on the 26B.
`VisionTower.encode_png` memoises soft tokens by PNG sha256 (LRU, 16).

**Check against upstream**: on a 160 x 192 gridworld frame the vendored
preprocess and tower reproduce mlx-vlm 0.7.8 **bit for bit** (pixel values
max |d| 0.0, projected soft tokens `[270, 1536]` max |d| 0.0; ours on mlx
0.32.0, mlx-vlm on mlx 0.32.4). The 26B tower loads and gives `[270, 2816]`.
Encoder time per frame, low power mode **on** (`pmset lowpowermode 1`): E2B
370 to 397 ms (mlx-vlm itself: 369 to 372 ms), 26B 1268 to 1282 ms.

## Licence check

- mlx-vlm: **MIT** (Copyright 2025 Prince Canuma). Compatible with this repo's
  MIT licence; the notice ships as `vision/LICENSE.mlx-vlm` and every vendored
  file names it.
- mlx-lm (the layer loop in `merge.py`): MIT, Apple; already a dependency.
- The processor's resize arithmetic is, per mlx-vlm, adapted from Hugging Face
  transformers' Gemma 4 processor (Apache-2.0); a six-line size formula,
  attributed in `images.resized_size` and `vision/processor.py`. Nothing else
  from transformers is copied, and transformers is not imported by the vision path.
- New runtime dependency: Pillow (HPND licence, permissive), behind the
  `vision` extra; imported lazily inside `preprocess_png`.

## Acceptance Criteria
- **Given** the package installed without mlx-vlm (import of `mlx_vlm` made to fail) **when** the vision path imports and encodes a PNG **then** it works. *(test_vision_path_runs_with_mlx_vlm_unimportable)*
- **Given** a text-only process **when** the runner, decide and server modules import **then** no `mlx_classifiers.vision`, `mlx_vlm` or PIL module is loaded. *(test_text_only_runner_import_loads_nothing_vision_related)*
- **Given** the E2B checkpoint **when** the tower loads **then** every vision weight is loaded strictly (clip bounds from the checkpoint, only the projector quantized) and a frame encodes to 270 x 1536 soft tokens, identically twice. *(test_tower_loads_every_vision_weight_and_quantizes_only_the_projector, test_encode_a_frame_gives_270_soft_tokens_deterministically)*
- **Given** any vendored file **then** it carries a provenance header (source, version, sha256, licence) and no source file imports mlx-vlm. *(test_vendored_files_carry_provenance_headers, test_no_module_imports_mlx_vlm)*
