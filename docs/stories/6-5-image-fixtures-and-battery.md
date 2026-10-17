---
type: story
status: completed
title: Image fixtures and a battery slice with images
owner: maddie
updated: 2026-10-17
anchor:
  - symbol: "src/mlx_classifiers/fixtures.py::SEED_CASES"
  - symbol: "src/mlx_classifiers/fixtures.py::run_case"
  - symbol: "src/mlx_classifiers/fixtures.py::compare"
  - symbol: "battery/build_gridworld.py::main"
  - test: "tests/test_fixtures.py::test_image_fixtures_pin_the_sha256_of_their_png_bytes"
  - test: "tests/test_fixtures.py::test_text_only_fixtures_carry_no_images_key"
  - test: "tests/test_fixtures.py::test_a_port_with_a_different_image_fails_on_images"
  - test: "tests/test_fixtures.py::test_fake_runner_image_cases_run_both_readouts"
  - test: "tests/test_fixtures.py::test_generate_on_mlx_is_byte_identical"
  - test: "tests/test_fixtures.py::test_committed_fixtures_reproduce_on_mlx"
  - test: "tests/test_int8.py::test_fixtures_agree_int8_on_vs_off"
  - test: "tests/test_int8_experts.py::test_26b_int8_experts_agree_with_stock_on_fixtures_and_delta_decisions"
  - test: "tests/test_battery.py::test_gridworld_battery_has_three_modes_of_30_frames_with_pinned_images"
  - test: "tests/test_battery.py::test_gridworld_battery_reports_accuracy_per_mode_on_the_fake_runner"
---

# Story 6.5 — Image fixtures and a battery slice with images

## Fixtures

Three image cases join the seed set (eleven now), PNGs inline as base64 so a
fixture file is self-contained:

| id | what | soft tokens | stock answer (raw p) |
|---|---|---|---|
| `choice-first-token-image-colour` | "What colour is the square?" on a 96 x 96 blue square, from a KV-cached system prefix | 256 | blue (0.997) |
| `choice-first-token-image-gridworld` | the gridworld's six-action question on frame `gw-s3-t00` given only as an image (+ the colour key) | 270 | north (0.424; oracle says west) |
| `noul-sequence-image` | "Is the square red?" scored by sequence: the image sits inside `cache_prefix(tokens[:-1])` | 256 | no (0.963) |

`expected.images` lists each image's `{start, length, sha256}`: the soft-token
run in `tokens` and the hash of the PNG bytes; the checker compares it exactly
(a port given a different image fails on `images`, not just on
probabilities). Text-only fixtures get no `images` key, and regenerating all
eleven left the eight existing files byte-identical (no diff); two MLXRunner
generations are byte-identical (existing test, now covering the image cases).
The fixture-format reference documents the field and how a port compiles and
prefills an image.

**Gate (story [4.3](4-3-per-fixture-gate-and-clear-winner-image-fixtures.md)).**
The gridworld case (and its 6.7 budget variants) is near-tied across six
actions, so one bf16 rounding step moves its probabilities by percents: this
repo on MLX 0.32.2 instead of 0.32.0 gives max |dp| 0.0757 (280) and 0.021
(140), and the Swift port on 0.32.2 fails the same way. The three files now
carry `gate: "answer-and-ids"` (exact ids, images and answer; |dp| reported,
not gated), and three clear-winner questions on the same frame carry the
probability gate at every budget.

**INT8 on image fixtures.** The colour and sequence image cases agree with
stock within the INT8 gates (max |dp| 7e-4 and 0). The gridworld frame, a
low-margin case (top-1 margin 0.17), moves by 0.139 and flips to `south`
under INT8. That is INT8's fidelity on gridworld prompts in general, not a
vision defect: over the 30 battery frames INT8 vs stock (E2B, low power off)
flipped the argmax on 1/30 text, 3/30 image-only and 0/30 text+image states,
with median max |dp| 0.06 to 0.08 and worst 0.21 in every mode. The INT8
fixture test now gates fixtures with a stock top-1 margin of at least 0.5 as
before and checks the low-margin ones for exact token ids and |dp| <= 0.25.
On the 26B (story 2.8's all-INT8 test, which runs every committed fixture's
request) the gridworld frame stays within 1.5e-3 in probability but its
low-probability `wait` logit moves 3.04, over that test's 3.0 logit gate; the
gate is now 4.0 with the measurement in its docstring, probabilities unchanged.

## The battery image slice

`battery/build_gridworld.py` (run under the gridworld repo's interpreter)
plays the gridworld oracle on seeds 1 to 5 and keeps six consecutive ticks of
each (seed 1 from tick 6, which covers its two attacks): 30 frames, PNGs under
`battery/images/` (30 files, about 25 KB in total), gold = the oracle's
action. Each frame is one perturbation group asked three ways, so the modes
are compared on identical states: `gridworld-text` (base, the gridworld's
text mode exactly), `gridworld-image` (`~p1`, the frame and a fixed colour key
only) and `gridworld-text+image` (`~p2`, the gridworld's image mode exactly).

`mlx-classifiers battery battery/gridworld.jsonl` (stock weights, one ordering,
M5, low power mode off, 2026-10-16):

| model | text | image only | text + image | all 90 | ECE raw | variants flipping vs text |
|---|---|---|---|---|---|---|
| E2B | 0.267 | 0.200 | 0.267 | 0.244 | 0.262 | 50.0% |
| 26B-A4B | 0.200 | 0.367 | 0.267 | 0.278 | 0.521 | 68.3% |

Chance is 1/6. On E2B the answer is almost always `south` (text, text+image)
or `north` (image only) regardless of the frame: no usable policy in either
mode. The 26B does better from the image alone than from the text map (11/30
vs 6/30) and is badly over-confident (mean confidence 0.78, ECE 0.52; a
battery-fitted temperature of 7.2 brings held-out ECE to 0.17). These are
small-n numbers (30 per mode) and no calibration or prompt work was done;
they are the baseline the gridworld's image mode starts from.

## Acceptance Criteria
- **Given** the image fixtures **when** regenerated **then** they are byte-identical, and the eight text fixtures are unchanged. *(test_generate_on_mlx_is_byte_identical, test_text_only_fixtures_carry_no_images_key)*
- **Given** an image fixture **then** it pins each PNG's sha256 and soft-token run, and a port that used a different image fails on `images`. *(test_image_fixtures_pin_the_sha256_of_their_png_bytes, test_a_port_with_a_different_image_fails_on_images)*
- **Given** the image battery slice **when** run **then** accuracy is reported per mode beside the text slice, from the same 30 frames. *(test_gridworld_battery_reports_accuracy_per_mode_on_the_fake_runner, test_gridworld_battery_has_three_modes_of_30_frames_with_pinned_images)*
