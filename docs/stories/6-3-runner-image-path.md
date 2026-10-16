---
type: story
status: completed
title: MLXRunner image path — encode, merge embeddings, same split prefill
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.prefill"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.cache_prefix"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.load_vision"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner._rows"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner._image_features"
  - symbol: "src/mlx_classifiers/mlx_runner.py::_bounds_off_runs"
  - symbol: "src/mlx_classifiers/vision/merge.py::forward_embeddings"
  - symbol: "src/mlx_classifiers/vision_bench.py::run_vision_bench"
  - test: "tests/test_vision_mlx.py::test_text_only_decide_never_loads_the_vision_tower"
  - test: "tests/test_vision_mlx.py::test_the_model_sees_the_image"
  - test: "tests/test_vision_mlx.py::test_image_request_is_deterministic"
  - test: "tests/test_vision_mlx.py::test_embedding_path_on_text_rows_equals_the_token_id_path"
  - test: "tests/test_vision_mlx.py::test_cache_only_chunks_split_inside_an_image_agree"
  - test: "tests/test_vision_mlx.py::test_delta_cache_keys_on_the_image_and_matches_uncached"
  - test: "tests/test_vision_mlx.py::test_int8_on_and_off_agree_on_image_requests"
  - test: "tests/test_images.py::test_chunk_bounds_move_off_image_runs_for_bidirectional_models"
---

# Story 6.3 — MLXRunner image path

`MLXRunner.prefill(tokens, prefix_cache, positions, images=slots)` (and
`cache_prefix(..., images=...)`) now run image requests on the same split
prefill as text (story 2.5): cache-only rows, then the readout rows only, then
the head on the readout rows only.

## How

- **Which rows go through embeddings.** `_rows(start, end)` is the backbone over
  one row range. With no image overlapping the range it is the old call on
  token ids, unchanged (text-only prefills are byte-for-byte the old path).
  Otherwise it builds `[1, L, D]` input embeddings, `embed_tokens(ids) *
  embed_scale` with the image's soft tokens in its rows (`vision.merge.merged_embeddings`,
  mlx-vlm's merge rule), per-layer token ids with the image rows set to the
  pad id 0 (as Gemma 4 / mlx-vlm do; the brief's "from the token ids as
  gemma4_text does" holds for every text row), and runs
  `vision.merge.forward_embeddings`: mlx-lm 0.31.3's `Gemma4TextModel.__call__`
  layer loop from the per-layer projection on, taking scaled embeddings. It
  calls the same layer modules, so the KV sharing (layers 15..34 reuse 0..14),
  the INT8 W8A8 / W4A8 patches and the MoE kernel apply unchanged. On text rows
  it is bit-identical to the token-id path (test).
- **Encoding.** The tower (story 6.2) loads on the first image
  (`load_vision`; `MLXRunner.load(..., vision=True)` preloads). Only images that
  overlap uncached rows are encoded, once per PNG sha256 (LRU of 16 in the
  tower); `vision_encode_ms` records each encode. A text-only process never
  loads it (test).
- **Bidirectional image blocks.** The 26B-A4B's text config has
  `use_bidirectional_attention: "vision"` (E2B: null): image rows attend to every
  row of their own image. `forward_embeddings` then builds array masks and ORs
  in the same-block matrix over the segment's columns (mlx-vlm's overlay), and
  cache-only chunk boundaries are moved off image runs (`_bounds_off_runs`). On
  E2B a boundary inside an image is exact up to chunking noise (test).
- **Delta cache.** Lookup and insert key on `images.keyed_tokens` (story 6.1),
  so the same placeholder ids with a different image share KV only up to the
  image's first soft token; a hit that would land inside an image run is
  retried capped at the run's start. A readout inside or before an image run is
  refused. A prefix cache built with images remembers them.

## Benchmarks

`python -m mlx_classifiers.vision_bench [--model R] [--int8]`, gridworld frames
(160 x 192, 270 soft tokens) from `battery/gridworld.jsonl` (story 6.5), the
gridworld's six-option action question, end-to-end `decide` per decision. M5,
AC power, mlx 0.32.0, mlx-lm 0.31.3. `single new`: a new frame every decision
(encodes each time); `single same`: one memoised frame (language model only);
`transcript`: `[frame, "Answer: a", frame, ...]` growing one tick per decision
over six consecutive ticks, only the newest frame new, old soft tokens
memoised in both rows, so uncached vs delta is KV reuse alone (median over 4
episodes, a fifth warms up). `encoder ms` is inside `p50 ms`.

### E2B, low power mode **off** (`lowpowermode 0` before and after), stock / INT8

| cell | frames | prompt tok | prefilled tok | p50 ms stock | p50 ms INT8 | encoder ms | decisions/s stock | decisions/s INT8 |
|---|---|---|---|---|---|---|---|---|
| single new | 1 | 510 | 457 | 310 | 293 | 198 | 3.23 | 3.41 |
| single same | 1 | 510 | 457 | 111 | 95 | 0 | 9.02 | 10.50 |
| transcript uncached | 1 | 446 | 393 | 299 | 300 | 198 | 3.34 | 3.33 |
| transcript uncached | 3 | 1014 | 961 | 379 | 351 | 199 | 2.64 | 2.85 |
| transcript uncached | 6 | 1868 | 1815 | 530 | 495 | 199 | 1.89 | 2.02 |
| transcript delta | 1 | 446 | 393 | 299 | 289 | 198 | 3.34 | 3.46 |
| transcript delta | 3 | 1014 | 394 | 288 | 277 | 200 | 3.47 | 3.61 |
| transcript delta | 6 | 1868 | 396 | 293 | 282 | 198 | 3.42 | 3.54 |

Peak MLX memory 4.3 GB (stock), 6.0 GB (INT8); no swapouts.

### 26B-A4B `--int8`, low power mode **off** (`lowpowermode 0` before and after)

| cell | frames | prompt tok | prefilled tok | cached tok | p50 ms | encoder ms | decisions/s |
|---|---|---|---|---|---|---|---|
| single new | 1 | 514 | 461 | 53 | 1251 | 675 | 0.80 |
| single same | 1 | 514 | 461 | 53 | 587 | 0 | 1.70 |
| transcript uncached | 1 | 450 | 397 | 53 | 1344 | 748 | 0.74 |
| transcript uncached | 2 | 734 | 681 | 53 | 1520 | 710 | 0.66 |
| transcript uncached | 3 | 1018 | 965 | 53 | 1756 | 720 | 0.57 |
| transcript uncached | 4 | 1302 | 1249 | 53 | 1998 | 707 | 0.50 |
| transcript uncached | 5 | 1586 | 1533 | 53 | 2381 | 719 | 0.42 |
| transcript uncached | 6 | 1872 | 1819 | 53 | 2817 | 712 | 0.35 |
| transcript delta | 1 | 450 | 397 | 53 | 1416 | 783 | 0.71 |
| transcript delta | 2 | 734 | 398 | 336 | 1174 | 736 | 0.85 |
| transcript delta | 3 | 1018 | 398 | 620 | 1212 | 752 | 0.82 |
| transcript delta | 4 | 1302 | 398 | 904 | 1203 | 734 | 0.83 |
| transcript delta | 5 | 1586 | 398 | 1188 | 1208 | 741 | 0.83 |
| transcript delta | 6 | 1872 | 400 | 1472 | 1202 | 715 | 0.83 |

Peak MLX memory 19.3 GB; about 59k vm swapouts during the run (32 GB machine).

### Earlier runs in low power mode **on** (`lowpowermode 1`)

E2B stock: single new 593 ms (encoder 374), single same 213 ms, transcript
delta flat at 546 to 562 ms from depth 2, uncached 573 ms at depth 1 rising to
1064 ms at depth 6. E2B INT8: single new 556, single same 175, delta 512 to
525, uncached 542 to 946 ms. A 26B `--int8` run in low power mode hit heavy
swapping (10.5 GB swap in use afterwards) and its transcript-uncached cells
read 9.6 to 14.2 s; it is not a valid number and is superseded by the table above.

### What the numbers say

- **The encoder is the floor.** A new 270-token frame costs 198 ms of vision
  tower on E2B (two thirds of a single-image decision) and 0.68 to 0.75 s on
  the 26B (16 vs 27 ViT layers over 2430 patches). Same-frame decisions show
  the language-model cost alone: 111 ms (E2B), 587 ms (26B, INT8).
- **Delta caching makes a frame transcript flat.** With only the newest frame
  new, every delta step prefills about 395 tokens (answer line, frame, question)
  whatever the depth; uncached grows with the transcript (E2B 299 -> 530 ms,
  26B 1.34 -> 2.82 s from 1 to 6 frames). At depth 6 delta is 1.8x (E2B) and
  2.3x (26B) the uncached rate, and the ratio keeps growing with depth.
- INT8 trims 10 to 16% off the language-model part on E2B; the tower is not
  INT8 (it is bf16, outside the patched decoder layers).
- The next lever is the tower: per-frame features could be computed ahead of
  the decision (the gridworld knows the frame before it asks), or a smaller
  `max_soft_tokens` budget (70 / 140 are processor-supported) would cut both
  the tower and the prefill; neither is in scope here.

## Acceptance Criteria
- **Given** an image request **when** run twice (re-encoding the image each time) **then** raw probabilities are identical. *(test_image_request_is_deterministic)*
- **Given** INT8 on and off **then** argmax agrees and raw probabilities are within 5e-3 on image requests. *(test_int8_on_and_off_agree_on_image_requests)*
- **Given** a coloured-square image **when** asked its colour **then** the model answers it (the image reaches the model). *(test_the_model_sees_the_image)*
- **Given** two prompts differing only in the image **when** the delta cache is on **then** the hit stops at the image's first soft token and the answer matches the uncached one; the same image again caches everything before the readout. *(test_delta_cache_keys_on_the_image_and_matches_uncached)*
- **Given** text-only rows **when** run through the embedding path **then** the hidden states equal the token-id path bit for bit; a text-only decide never loads the tower. *(test_embedding_path_on_text_rows_equals_the_token_id_path, test_text_only_decide_never_loads_the_vision_tower)*
