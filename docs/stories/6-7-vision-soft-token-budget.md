---
type: story
status: completed
title: Vision soft-token budget — 70 / 140 / 280 per request, gated on the battery and the loop
owner: maddie
updated: 2026-10-17
anchor:
  - symbol: "src/mlx_classifiers/images.py::SOFT_TOKEN_BUDGETS"
  - symbol: "src/mlx_classifiers/images.py::image_key_id"
  - symbol: "src/mlx_classifiers/images.py::ImageSlot"
  - symbol: "src/mlx_classifiers/schema.py::RequestOptions"
  - symbol: "src/mlx_classifiers/prompt.py::compile_question"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.vision_soft_tokens"
  - symbol: "src/mlx_classifiers/vision/tower.py::VisionTower.encode_png"
  - symbol: "src/mlx_classifiers/vision/fast.py::FastEncoder.uint8"
  - symbol: "src/mlx_classifiers/fixtures.py::SEED_CASES"
  - test: "tests/test_vision_budget.py::test_the_gridworld_frame_resizes_and_pools_per_budget"
  - test: "tests/test_vision_budget.py::test_the_wire_option_takes_the_three_supported_budgets_only"
  - test: "tests/test_vision_budget.py::test_a_request_budget_sets_the_run_length_and_rides_on_the_slot"
  - test: "tests/test_vision_budget.py::test_the_same_png_at_two_budgets_never_shares_a_delta_key"
  - test: "tests/test_vision_budget.py::test_delta_cache_hit_stops_at_the_image_when_only_the_budget_changes"
  - test: "tests/test_vision_budget.py::test_tower_encodes_each_budget_to_its_own_count_and_memoises_per_budget"
  - test: "tests/test_vision_budget.py::test_runner_budget_field_validates_and_defaults_to_the_checkpoint"
  - test: "tests/test_vision_mlx.py::test_vision_fast_is_bit_equal_at_each_soft_token_budget"
  - test: "tests/test_vision_parity.py::test_merged_embeddings_and_readout_match_mlx_vlm"
  - test: "tests/test_fixtures.py::test_image_fixtures_pin_the_sha256_of_their_png_bytes"
  - test: "tests/test_int8_experts.py::test_26b_int8_experts_agree_with_stock_on_fixtures_and_delta_decisions"
---

# Story 6.7 — Vision soft-token budget

Story 6.6 left the 26B encode at 598 ms per frame and ranked the levers for
3x. This story built the first one, the soft-token budget, and gated it. The
second, SDPA width 128, is not built: it moves to story
[6.8](6-8-vision-sdpa-width.md), because the budget did not pass its gate as a
default.

## What a budget is

`max_soft_tokens` is a ceiling, not a count. Gemma 4's processor resizes the
image, aspect kept, to the largest size whose sides are multiples of 48 px and
that holds at most `budget x 9` patches of 16 px. The tower runs over those
patches, then pools them 3 x 3, so an image gets at most `budget` soft tokens.
For a 160 x 192 gridworld frame:

| budget | resize | patches (tower length) | soft tokens | 26B tower ms (6.6 profile) |
|---|---|---|---|---|
| 280 (default) | 720 x 864 | 2430 | 270 | 590 |
| 140 | 480 x 576 | 1080 | 120 | 236 |
| 70 | 336 x 432 | 567 | 63 | 105 |

The tower's cost follows the patch count, and attention is quadratic in it,
so it falls faster than the token count. The prompt is also shorter by the
difference in soft tokens (150 fewer at 140).

## As built

- **Wire option** `options.vision_soft_tokens`: 70, 140 or 280, optional but
  not nullable (schema file and pydantic agree). Omitted, the runner's budget
  applies. Gemma 4 also allows 560 and 1120; this library supports the three
  measured here (`images.SOFT_TOKEN_BUDGETS`).
- **Runner level** `MLXRunner(vision_soft_tokens=N)` / `MLXRunner.load(...,
  vision_soft_tokens=N)` rewrites `image_tokens.max_soft_tokens`, so prompts
  compile to the budget's run length. `serve`, `battery` and `vision_bench`
  take `--vision-tokens N`; `battery` also gained `--int8`. Both
  reports record the budget.
- **Slot and cache key.** `compile_question` records the budget on each
  `ImageSlot` (`max_soft_tokens`). The runner encodes at the slot's budget.
  The delta-cache key mixes the budget into the image's stand-in id
  (`image_key_id(sha256, budget)`), so the same PNG at two budgets never shares
  KV. Without that, a 140 run would reuse the first 120 rows of a 280 run.
  `VisionTower` memoises features by `(sha256, budget)`. `FastEncoder` pins
  one compiled shape per budget.
- **Fixtures.** The default did not change, so the eleven fixtures were
  regenerated and came out byte-identical (no diff). Two new cases,
  `choice-first-token-image-gridworld-vt140` and `-vt70`, carry the budget in
  the request and in `expected.images[].max_soft_tokens`. `describe()` omits
  the key at 280, so the default fixtures keep their exact images record.
  `fixtures check --runner mlx --logit-tol 0`: 13/13, max |dlogit| 0.
  Since story [4.3](4-3-per-fixture-gate-and-clear-winner-image-fixtures.md)
  these two are answer-and-ids checks (near-tied), and
  `choice-first-token-image-gridworld-you-vt140` / `-vt70` carry the
  budget path under the full gate.

## Measured (M5, AC, low power mode off before and after every run; mlx 0.32.0)

Raw output is in `bench_runs/6-7/` (untracked). It includes `runs.log` with
power state and vm swapouts before and after each run. Runs were sequential,
one model loaded at a time, a fresh process per run.

### (a) Gridworld battery, 30 frames per mode

Text-only accuracy does not depend on the budget: E2B 0.267, 26B INT8 0.167,
26B stock 0.200 at every budget.

| model | budget | image only | text + image | swapouts |
|---|---|---|---|---|
| E2B stock | 280 | 0.200 | 0.267 | 0 |
| E2B stock | 140 | 0.267 | 0.267 | 0 |
| E2B stock | 70 | 0.233 | 0.267 | 0 |
| 26B `--int8` | 280 | 0.333 | 0.300 | 97k |
| 26B `--int8` | 140 | 0.267 | **0.067** | 82k |
| 26B `--int8` | 70 | 0.233 | 0.100 | 18k |
| 26B stock | 280 | 0.367 | 0.267 | 2k |
| 26B stock | 140 | 0.200 | 0.200 | 22k |
| 26B stock | 70 | 0.300 | 0.100 | 32k |

E2B answers almost one action per mode at every budget (image only: north
26 to 30 of 30; text + image: south 28 to 30 of 30). Its accuracy is the base
rate of that action, so it says nothing about the budget. The 26B is the
model that reads the frame. At 140 and 70 its text + image answers collapse
toward `attack` (16 of 30 at 280, 22 at 140, 21 at 70 under INT8). Stock
weights show the same drop, so the cause is the budget, not INT8. 26B stock at
280 reproduces story 6.5's row exactly. Swapping does not change numerics,
only time.

### (b) The gridworld loop, the sequential test

The sibling's winning configuration, `gridworld play --backend local --model
mlx-community/gemma-4-26B-A4B-it-qat-4bit --int8 --labels words --legal-only
--memory summary --mode image --episodes 5 --seed 0 --vision-tokens N` (new
flag, gridworld story 3.5 follow-up). Run dirs are under the gridworld's
`runs/story-3-5-vt/`.

| budget | won | reversals / ep | ticks | p50 ms | p50 encoder ms | decisions/s | swapouts |
|---|---|---|---|---|---|---|---|
| 280 (story 5.4 row) | 5/5 | 12.6 | 154 | 1290 | 709 | - | - |
| 280 (rerun, same code) | 5/5 | 12.6 | 154 | 1442 | 766 | 0.64 | 109k |
| 140 | 5/5 | **9.6** | 117 | 666 | 221 | 1.46 | 11k |
| 70 | **4/5** | 25.8 | 255 | 633 | 141 | 1.55 | 151k |

The 280 rerun matches the 5.4 row seed for seed (seed 4 won in 99 steps
with 62 reversals). That confirms the 6.6 fast path plays identically. Its
timings are swamped (109k swapouts). At 140, seed 4 won in 39 steps
(23 reversals) and seed 1 took 36 (23 reversals; 13 and 0 at 280). At 70,
seed 4 timed out at 200 ticks with 129 reversals.

### Encoder ms and decisions/s (`vision_bench --vision-tokens N`)

| model | budget | prompt tok | encoder ms | p50 ms | decisions/s single new | decisions/s transcript, 6 frames | swapouts |
|---|---|---|---|---|---|---|---|
| E2B | 280 | 510 | 139 | 247 | 4.04 | 4.30 | 0 |
| E2B | 140 | 360 | 60 | 146 | 6.86 | 8.10 | 0 |
| E2B | 70 | 303 | 32 | 108 | 9.29 | 11.76 | 0 |
| 26B `--int8` | 280 | 514 | 620 | 1202 | 0.83 | 0.82 | 48k |
| 26B `--int8` | 140 | 364 | 263 | 818 | 1.22 | 1.65 | 14k |
| 26B `--int8` | 70 | 307 | 136 | 648 | 1.54 | 2.40 | 0 |

The 26B's 280 and 140 rows swapped, so treat their p50s as upper bounds.
The 70 row did not swap.

### Parity

- `vision_fast` vs the vendored path at 140 and 70 on E2B, GPU: bit-equal
  (`test_vision_fast_is_bit_equal_at_each_soft_token_budget`).
- Against mlx-vlm 0.7.8 at each budget (`test_vision_parity`, parametrized
  over 280 / 140 / 70; reference processor called with the same
  `max_soft_tokens`). Pixels are exact at every budget, and so are soft tokens
  at 280. At 140 and 70 the soft tokens differ by bf16 rounding (max |d| 0.0156
  on E2B, 0 / 0.0625 on the 26B), and the option logits by up to 0.125 on E2B
  and 0.25 / 0.44 on the 26B. Argmax agrees everywhere. The cause is the mlx
  version, not the code. In the mlx-vlm venv (mlx 0.32.4) our vendored tower
  equals mlx-vlm's bit for bit at all three budgets on both models. This
  repo pins mlx 0.32.0. Off-default budgets are therefore gated at
  `OFF_DEFAULT_FEAT_TOL` 0.0625 and `OFF_DEFAULT_LOGIT_TOL` 0.5, with the
  measurements in the test.
- Found in passing: on mlx 0.32.4 the 26B's **compiled** fast path is not
  bit-equal to the vendored path (max |d| 5.0 at 280, 1.4 at 140). It is
  bit-equal on the pinned 0.32.0 (6.6), and E2B is bit-equal on both. An mlx
  upgrade must re-run 6.6's GPU bit-equality before moving the pin.
- 26B INT8 experts vs stock on the budget fixtures: max |dp| 0.196 (140) and
  0.154 (70), against 0.0015 at 280; argmax unchanged. These two fixtures are
  gated at |dp| 0.25 in `test_int8_experts` (`BUDGET_PROB_TOL`).

## Decision: the default stays 280

The rule: the smallest budget whose battery accuracy is within 2 points of
280 and whose 5-seed completion is unchanged. On the 26B, 140 keeps
completion (5/5, fewer reversals) but drops 23 points on text + image under
INT8 and 17 points on image only under stock weights. That fails the battery
gate. 70 fails both gates. E2B passes at every budget, but only because it
does not read the frame, so it cannot carry the decision. **280 stays the
default.** The image fixtures are not regenerated, and the non-default
budgets are pinned by their own fixtures.

140 is now a measured opt-in: `--vision-tokens 140` in the gridworld, or the
wire option. The loop plays it at 2.3x the decisions/s (1.46 against 0.64
swamped, or about 0.78 unswamped from 5.4's 1.29 s p50), with no loss in
completion. The battery says the frame is read less well. With n = 5 seeds
and n = 30 frames, both signals are small. Settling the conflict needs more
seeds, not this story.

## Acceptance Criteria
- **Given** `options.vision_soft_tokens` or `MLXRunner(vision_soft_tokens=N)`
  **when** a request with an image compiles **then** the run is the budget's
  soft-token count, the slot records the budget, the tower encodes at it, and
  the same PNG at two budgets shares no cached KV. *(test_a_request_budget_sets_the_run_length_and_rides_on_the_slot, test_the_same_png_at_two_budgets_never_shares_a_delta_key, test_delta_cache_hit_stops_at_the_image_when_only_the_budget_changes, test_tower_encodes_each_budget_to_its_own_count_and_memoises_per_budget)*
- **Given** budgets 140 and 70 **then** the fast path is bit-equal to the
  vendored path on the GPU, and mlx-vlm parity holds within the off-default
  tolerances, same argmax. *(test_vision_fast_is_bit_equal_at_each_soft_token_budget, test_merged_embeddings_and_readout_match_mlx_vlm)*
- **Given** the battery at 280 / 140 / 70 on E2B and the 26B, and the 26B
  gridworld loop at 280 / 140 / 70 **then** the default is the smallest budget
  within 2 points of 280 with unchanged completion. Measured: 280 (tables above).
- **Given** the default unchanged **then** the eleven fixtures regenerate
  byte-identically, and one fixture per non-default budget records its budget.
  *(test_image_fixtures_pin_the_sha256_of_their_png_bytes)*
- The outline's "encoder at most a third of 659 ms on the 26B" is met only at
  a non-default budget (70: 136 ms; 140: 263 ms bench, 221 ms loop p50). At
  the default it moves to story 6.8.
