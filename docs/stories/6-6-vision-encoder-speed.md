---
type: story
status: completed
title: Vision encoder speed — 675 ms per frame on the 26B is not the hardware
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/vision/fast.py::FastEncoder"
  - symbol: "src/mlx_classifiers/vision/processor.py::resized_uint8"
  - symbol: "src/mlx_classifiers/vision/tower.py::VisionTower.fast_encoder"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.load_vision"
  - test: "tests/test_vision_fast.py::test_fast_uncompiled_equals_slow_bit_for_bit"
  - test: "tests/test_vision_fast.py::test_fast_compiled_equals_slow_bit_for_bit"
  - test: "tests/test_vision_fast.py::test_position_embedding_gather_equals_the_one_hot_matmul"
  - test: "tests/test_vision_fast.py::test_uint8_lookup_patchify_equals_the_float32_patchify"
  - test: "tests/test_vision_fast.py::test_one_trace_per_pinned_shape"
  - test: "tests/test_vision_fast.py::test_runner_vision_fast_defaults_on_and_the_tower_keeps_the_slow_reference"
  - test: "tests/test_vision_mlx.py::test_vision_fast_is_bit_equal_to_the_vendored_path"
  - symbol: "src/mlx_classifiers/vision_bench.py::main"
---

# Story 6.6 — Vision encoder speed

Story 6.3's bench (low power off) measured the encode of one 160x192
gridworld frame at 198 ms on E2B and 675 ms on the 26B. With delta caching
making frame history free, per-tick cost in image mode is roughly encoder
plus ~395 prefilled tokens, so the encoder is now the largest single cost in
the vision path. Those numbers are far above what a tower of this size
should take on an M5 at full power, which points at the vendored path rather
than the model: dtype wider than needed, no `mx.compile` on the tower or
projector, Python-side work in preprocessing, or kernel recompiles because
shapes are not pinned.

Profile first (preprocess, tower, projector, merge, each behind
`mx.synchronize`), then fix the largest, in order of likelihood above.
Parity gate: the image fixtures and the mlx-vlm parity test (6.4) must still
pass within their tolerances.

## Inspection (before profiling)

Done by reading only (the GPU was busy): no model loaded, nothing timed. Line
numbers are as of commit `3cba682` (the vendored path before this story). The
frame is 160 x 192, resized to 720 x 864 = 45 x 54 = **2430 patches**, pooled
3 x 3 to 270 soft tokens.

**The FLOP floor comes first.** Counting matmuls only (projections, MLP,
attention at the fused-SDPA head width), the encoder layers are **1.02 TFLOP**
per frame on E2B (16 layers, hidden 768) and **3.47 TFLOP** on the 26B (27
layers, hidden 1152, head_dim 72 padded to 80), plus 76 / 115 GFLOP for the
position embedding (cause 1). 198 ms and 675 ms are then 5.6 and 5.3 TFLOP/s:
the 26B is not anomalous next to E2B (time ratio 3.4, FLOP ratio 3.3), and the
whole tower runs at one efficiency. Whether that is far from the M5's bf16 peak
is what the profile's TFLOP/s line settles. If the matmuls already run near it,
the fixes below are worth tens of percent, not 3x, and the 3x gate needs one of
the larger levers at the end of this section. mlx-vlm 0.7.8 runs the same tower
uncompiled and with the same ops (6.2 measured it at our speed). mlx-lm only
compiles small elementwise helpers (`@partial(mx.compile, shapeless=True)` on
activations in `gemma3n.py`, `activations.py`), never a whole tower.

Ranked by expected cost, each checklist item as the code does it now:

1. **Position embedding is a one-hot matmul** (checklist 3). `encoder.py:288-293`
   `VisionPatchEmbedder._position_embeddings` builds `one_hot(positions, 10240)`
   in float32 (`[1, 2430, 2, 10240]`, 199 MB), casts it to bf16 (100 MB more) and
   multiplies by the `[2, 10240, D]` table: 76 GFLOP on E2B, 115 on the 26B, plus
   about 350 MB of transients per frame (on a 32 GB machine already swapping in
   the 26B bench). It is the same for every frame of a given size.
2. **Nothing is compiled, and every layer runs about 120 small kernels**
   (checklists 2 and 1). No `mx.compile` anywhere in `src/`. Per layer, the
   float32 RMS-norm round trips for q, k and v (`encoder.py:126-144`,
   `VisionRMSNorm` / `VisionRMSNormNoScale`: astype, square, mean, rsqrt, mul,
   astype), the 2D RoPE tables recomputed from the positions for q and for k in
   every layer (`encoder.py:182-196`: arange, power, divide, cos, sin, concat,
   cast, twice per spatial dim), the clips on E2B (`encoder.py:105-111`, two per
   linear, seven linears), GELU and residuals. All memory-bound on
   `[2430, D]`-sized tensors and unfused.
3. **An all-zero additive mask goes to SDPA in every layer** (checklist 5).
   `encoder.py:421-425` builds a bf16 `[1, 1, 2430, 2430]` mask; with
   `max_patches = num_real` (`encoder.py:413`) there is never padding, so it is
   all zeros, but SDPA reads it for every head of every layer. The attention
   itself is `mx.fast.scaled_dot_product_attention` (`encoder.py:78-85`), not a
   manual softmax. On the 26B, head_dim 72 is padded to 80 (`encoder.py:79-85`):
   three pads and a slice per layer.
4. **Host sync mid-pipeline** (checklist 8). `encoder.py:437`
   `int(valid_mask[i]...sum().item())` forces the whole tower before the
   projector graph is even built, then `tower.py:79` (`encode_png`'s `mx.eval`)
   runs the projector as a second graph. The count is always `L / 9` here. It
   also makes the forward uncompilable as written.
5. **Preprocess on the host** (checklist 3). `processor.py:39-50`: PIL decode,
   bicubic 4.5x upscale, then `astype(float32) * (1/255)` and a channel-first
   transpose in numpy (7.5 MB float32), copied by `mx.array` (`tower.py:65`).
   Per-call numpy `meshgrid` patch positions and two `mx.array` builds
   (`encoder.py:389-415`). No per-patch Python loops. A few ms at most.
6. **Pooler** (checklist 1). `encoder.py:318-330` builds the `[1, 2430, 270]`
   float32 one-hot weights per call and the einsum promotes the bf16 activations
   to float32 (about 1 GFLOP). Small; the weights are cacheable per shape.

The rest of the checklist, found not to be causes:

- **dtype** (1): the checkpoints are bf16 throughout the tower, clip bounds
  included (read from the safetensors headers), so the clipped linears stay bf16;
  only the projector is 4-bit. Float32 islands are pixels (`processor.py:49`),
  the q/k/v norms, the RoPE tables and the pooler einsum, all cast back.
- **shape pinning** (4): a frame size gives one shape every call, so there is
  nothing to recompile today; it matters only once compiled (one trace per shape).
- **merge** (6): `merge.py:31-45` `merged_embeddings` is one `mx.concatenate` of
  slices and the features, not a per-row scatter; `per_layer_token_ids` is one
  `mx.where`. Neither is inside `vision_encode_ms`. Left as is.
- **26B bidirectional mask** (7): `merge.py:59-94` (`block_ids` builds a Python
  list of length L; `_overlay` ORs an `[N, N]` same-block matrix into the
  mask). It belongs to the LM prefill, not the encoder. Its real cost is that
  image chunks get array masks instead of mlx-lm's causal fast path in every
  decoder layer, which the profile does not measure.
- **weight loading** (9): once per runner (`mlx_runner.py:310-319`
  `load_vision`), evaluated at load (`tower.py:156`), features memoised by PNG
  sha256. Nothing is re-read or re-cast per call beyond the per-layer
  `weight.astype(float32)` of the `[head_dim]` q/k norm scales.

## The fix: `vision_fast` (default on since this story)

`vision/fast.py::FastEncoder` runs the tower's own modules for the runner's one
case: one image, no padding. The vendored path is untouched and stays the
reference: `MLXRunner(vision_fast=False)`, or `vision_bench --vision-slow`.
`MLXRunner.vision_fast` (default **True**) → `load_vision_tower(fast=True)` →
`VisionTower.encode_png`. Every change is bit-exact:

- position embedding: `table[0][x] + table[1][y]`, a gather, cached per image
  shape;
- RoPE cos/sin, patch positions and pooling weights: cached per shape;
- `mask=None` in SDPA; no `.item()`;
- preprocess: PIL to uint8 (`processor.resized_uint8`, which `preprocess_png`
  now calls), uploaded as uint8; `x / 255` then `2 (x - 0.5)` become a 256-entry
  lookup table built with exactly those float32 ops;
- `mx.compile` per layer in **two segments**: (input norm, q/k/v projections)
  and (RoPE, SDPA, o_proj, post norm, residual, MLP, residual). Each is traced
  once per pinned shape. The float32 q/k/v RMS norms between the segments stay
  eager.

**Why the norms stay out of compile.** The first draft compiled the whole
forward. On the GPU it was 130 ms on E2B but **not bit-equal** (max |d| 0.11 in
the soft tokens). Bisecting by compiling one module class at a time: clipped
linears, MLP, the fast RMSNorm and RoPE all compile bit-exactly. The float32
`VisionRMSNorm` / `VisionRMSNormNoScale` do not. Isolated, eager `x ** 2` (Metal
`pow`) and the fused square differ by about 1 ulp in float32 (rel 1.2e-7), and
16 layers grow that to 0.08 to 0.11. Leaving the three norms eager costs about
5 ms on E2B and gives bit-equality.

`tests/test_vision_fast.py` checks every stage on CPU against a tiny random
tower. `tests/test_vision_mlx.py::test_vision_fast_is_bit_equal_to_the_vendored_path`
pins it on the GPU with the real E2B weights.

## Measured (M5, AC power, low power mode **off** before and after every run; mlx 0.32.0)

### Profile (`proto/vision_profile.py`, 10 frames after one warm-up, median ms)

Raw output is in `bench_runs/6-6/` (untracked). "LM" means the full runner is
resident (26B with `--int8`, peak 19.6 GB), as in the bench. Staged rows are
fenced with `mx.synchronize` after each stage. End-to-end rows are one graph.

| stage | E2B slow | E2B fast | 26B slow (LM) | 26B fast (LM) |
|---|---|---|---|---|
| preprocess (host) | 3.1 | 2.2 | 3.1 | 2.2 |
| upload | 0.4 | 0.1 | 0.3 | 0.1 |
| patch + position embed | 14.2 | 1.5 | 19.0 | 1.8 |
| mask | 0.4 | — | 0.4 | — |
| tower layers (uncompiled) | 175.2 | 170.7 | 636.8 | 624.3 |
| pool (+ `.item()` sync on slow) | 0.9 | 0.5 | 1.2 | 0.8 |
| projector | 0.3 | 0.3 | 0.5 | 0.5 |
| merge (embeddings + per-layer ids + block ids) | 0.4 | 0.4 | 4.0 | 4.0 |
| **end to end, one graph** | **193.2** | **138.1** (compiled) | **659.2** | **597.5** (compiled) |
| tower TFLOP/s (matmul count) | 5.3 | 7.4 | 5.3 | 5.8 |

Tower-only runs (no LM resident) agree: E2B 194.4 → 137.7 ms, 26B 653.5 →
590.2 ms. Fast path uncompiled: E2B 176.3 ms. Speed-up: **1.40x on E2B, 1.10x on
the 26B**.

Micro-benchmarks (best of 10):

- position embedding, one-hot matmul vs gather: E2B 11.9 vs 0.9 ms, 26B 15.3 vs
  1.2 ms; bit-equal.
- SDPA for one layer, with the all-zero mask vs none: E2B 1.89 vs 1.60 ms, 26B
  9.36 vs 8.90 ms; bit-equal.

**Where the 26B's time goes.** The fast 26B tower is two things.

- Its linears: about 12 ms per layer, 327 ms in all. bf16 runs at 11 TFLOP/s on
  the MLP shapes and 4.6 TFLOP/s on the 1152-square q/k/v/o.
- SDPA at head_dim 72, padded to 80: **9.35 ms per layer, 252 ms in all, 3.4
  TFLOP/s**. E2B's head_dim 64 runs at about 11 TFLOP/s.

`[1, 16, 2430, d]` SDPA by padded width: d=72 unpadded 8.1 ms, 80 9.35 ms, 96
8.3 ms, **128 5.2 ms**. None of these is bit-equal to 80; max |d| is 1e-3 per
call. A manual matmul + softmax takes 19.2 ms.

### Parity

- `vision_fast` vs vendored, GPU, bit-equal on every frame: E2B 11/11 (LM and
  tower-only), 26B 11/11 (LM and tower-only); max |d| 0.
- `pytest -m mlx tests/test_vision_mlx.py tests/test_vision_parity.py
  tests/test_battery_mlx.py` with `MLX_VLM_PYTHON` pointing at the mlx-vlm 0.7.8
  venv: 13 passed. That includes 6.4 against mlx-vlm on E2B and the 26B, through
  the default (fast) runner.
- `mlx-classifiers fixtures check fixtures/ --runner mlx --logit-tol 0`: 11/11,
  the image fixtures included, max |dp| 0, max |dlogit| 0.

### Image bench before / after (`python -m mlx_classifiers.vision_bench [--vision-slow]`)

| model | cell | encoder ms slow → fast | p50 ms slow → fast | decisions/s slow → fast |
|---|---|---|---|---|
| E2B stock | single new | 195 → 140 | 304 → 249 | 3.29 → 4.01 |
| E2B stock | transcript delta, 6 frames | 197 → 141 | 289 → 233 | 3.45 → 4.28 |
| 26B `--int8` | single new | 683, 811 → 723, 687 | 1266, 1518 → 1404, 1335 | 0.79, 0.66 → 0.71, 0.75 |
| 26B `--int8` | transcript delta, 6 frames | 793, 810 → 754, 730 | 1311, 1336 → 1271, 1241 | 0.76, 0.75 → 0.79, 0.81 |

E2B: no swap; peak 4.3 GB. The 26B was run twice per path (slow, fast, fast,
slow). Swapping took 20k to 96k vm swapouts per run. "single same" never
encodes, yet it moved 602 to 708 ms between runs, so in-bench 26B numbers carry
±60 ms of swap noise. The encode gain there, 40 to 60 ms, is within that noise.
The profile above is the clean 26B number.

**Acceptance against the outline.** The profile criterion is met. The 3x target
on the 26B is **not** met by bit-exact fixes: the tower is now mostly matmul
plus a slow SDPA width. The 3x moves to story 6.7, chosen from the data below.

### Data for the next lever (story 6.7)

- **INT8 on the tower linears: rejected.** The repo's W8A8 kernel at tower
  shapes (2430 rows) gave 26B q/k/v/o 1.41 → 1.12 ms, MLP gate/up 2.18 → 2.23 ms
  (the down projection could not be quantized: 4304 is not a multiple of 64),
  and E2B no better than bf16. bf16 already reaches 10 to 11 TFLOP/s on the MLP
  shapes. Rel error is about 1e-2.
- **SDPA width on the 26B**: pad 72 → 128 instead of 80 saves about 4.1 ms per
  layer, about 110 ms per frame (598 → about 490 ms). It is not bit-exact
  (1e-3 per call), so it needs the 6.4 tolerance gate rather than equality.
- **A smaller soft-token budget** (fast path, tower only):

  | budget | soft tokens | patches | E2B ms | 26B ms |
  |---|---|---|---|---|
  | 280 (today) | 270 | 2430 | 138 | 590 |
  | 140 | 120 | 1080 | 59 | 236 |
  | 70 | 63 | 567 | 31 | 105 |

  140 is 2.5x on the 26B tower and also prefills about 150 fewer tokens per
  frame. It changes what the model sees, so it needs regenerated image fixtures
  and the gridworld battery's accuracy compared.
- **Encode the next frame during the current decision**: one GPU, so this hides
  latency only where the GPU would otherwise idle between decisions. It does not
  raise decisions/s when decisions run back to back.

## Acceptance Criteria
- **Given** the profile **when** recorded **then** each stage has a measured
  share on both E2B and the 26B (table above).
- **Given** `vision_fast` (default) **when** compared to the vendored path on
  the GPU **then** the soft tokens are bit-equal on E2B and the 26B, the 6.4
  parity and the image fixtures pass unchanged (max |dlogit| 0), and encoder ms
  per new frame drops (E2B 1.40x, 26B 1.10x by profile). *(test_vision_fast_is_bit_equal_to_the_vendored_path,
  test_fast_compiled_equals_slow_bit_for_bit)*
- The outline's "at least 3x on the 26B" is not reachable bit-exactly. It moves
  to story 6.7.
