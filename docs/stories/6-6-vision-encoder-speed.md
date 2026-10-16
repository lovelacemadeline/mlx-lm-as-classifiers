---
type: story
status: in-progress
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
  - test: "tests/test_vision_fast.py::test_runner_vision_fast_defaults_off"
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

### Draft fix: `vision_fast` (default off)

`vision/fast.py::FastEncoder` runs the tower's own modules for the runner's one
case (one image, no padding). The vendored path is untouched and stays the
reference. `MLXRunner(vision_fast=True)` → `load_vision_tower(fast=True)` →
`VisionTower.encode_png` uses it. Changes, each meant to be bit-exact:

- position embedding: `table[0][x] + table[1][y]`, a gather, cached per image
  shape (a one-hot row times the table is exactly that row);
- RoPE cos/sin, patch positions and pooling weights: cached per shape and reused
  by every layer;
- `mask=None` in SDPA (adding exact zeros); no `.item()`;
- preprocess: PIL to uint8 (`processor.resized_uint8`, which `preprocess_png`
  now calls), uploaded as uint8; `x / 255` then `2 (x - 0.5)` become a 256-entry
  lookup table built with exactly those float32 ops, so the bf16 patch inputs
  are identical;
- `mx.compile` over the whole forward (embed, layers, pool, projector), traced
  once per pinned shape and cached (`compile=False` to turn it off).

`tests/test_vision_fast.py` (CPU device, a random 2-layer bf16 tower with a
4-bit projector, E2B-like / 26B-like / padded-head variants): every stage and
the end-to-end soft tokens are **bit-equal** to the slow path, compiled and not.
CPU equality does not prove GPU equality: SDPA with and without a mask, and the
fused compiled kernels, are different Metal code. The profile checks that.

**Larger levers if the profile shows the matmuls near peak** (not drafted):
the tower's 2430-row linears are the shape the W8A8 NAX kernel (story 2.3) is
best at; a 140-soft-token budget (`--soft-tokens 140`) roughly halves the linears
and quarters the attention, but changes what the model sees; or encode ahead of
the decision.

### To finish (GPU free; low power off: `sudo pmset -a lowpowermode 0`)

1. Profile both models, both paths, LM resident and not:

   ```bash
   python proto/vision_profile.py
   python proto/vision_profile.py --tower-only
   python proto/vision_profile.py --model mlx-community/gemma-4-26B-A4B-it-qat-4bit --int8
   python proto/vision_profile.py --model mlx-community/gemma-4-26B-A4B-it-qat-4bit --tower-only
   python proto/vision_profile.py --no-compile --path fast        # compile's share
   python proto/vision_profile.py --dtype float16 --path both     # dtype check
   ```

   Record each stage's share here, the TFLOP/s line, the two micro-benchmarks
   and the parity line (`bit-equal n/n` or `max |d|`).
2. If parity is bit-equal on both models (or within the 6.4 tolerances), flip
   the default: `vision_fast: bool = True` in `MLXRunner`, and run the gates:

   ```bash
   pytest -m mlx tests/test_vision_mlx.py tests/test_vision_parity.py tests/test_battery_mlx.py
   pytest tests/test_vision_fast.py
   mlx-classifiers fixtures check fixtures/ --runner mlx   # image fixtures unchanged (6.5)
   ```

   If it is not bit-equal, the two candidates are `mask=None` (the SDPA
   micro-benchmark prints its own bit-equality) and compile (compare the parity
   line of a `--no-compile` run); keep the one that differs off.
3. Re-bench and compare the `encoder ms` column against 198 / 675 ms:

   ```bash
   python -m mlx_classifiers.vision_bench
   python -m mlx_classifiers.vision_bench --model mlx-community/gemma-4-26B-A4B-it-qat-4bit --int8
   ```

   If the 26B is not 3x faster, take the levers above in order. That is the
   decision this story still has to make.

## Acceptance Criteria
- **Given** the profile **when** recorded **then** each stage has a measured
  share on both E2B and the 26B.
- **Given** the fixes **when** the image bench reruns **then** encoder ms per
  new frame drops by at least 3x on the 26B with image fixtures unchanged.
