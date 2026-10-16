---
type: story
status: completed
title: Battery runner — accuracy, ECE, position-bias report
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/battery.py::run_battery"
  - symbol: "src/mlx_classifiers/battery.py::run_case"
  - symbol: "src/mlx_classifiers/battery.py::summarise"
  - symbol: "src/mlx_classifiers/battery.py::calibration_records"
  - symbol: "src/mlx_classifiers/battery.py::render_markdown"
  - symbol: "src/mlx_classifiers/battery.py::write_report"
  - symbol: "src/mlx_classifiers/battery.py::main"
  - test: "tests/test_battery.py::test_battery_on_fake_runner_reports_every_metric_and_round_trips"
  - test: "tests/test_battery.py::test_accuracy_ece_and_position_bias_on_a_first_slot_runner"
  - test: "tests/test_battery.py::test_perturbation_flip_rate_counts_variants_that_change_answer"
  - test: "tests/test_battery.py::test_calibration_config_gives_calibrated_ece"
  - test: "tests/test_battery_mlx.py::test_battery_subset_on_mlx"
---

# Story 5.2 — Battery runner — accuracy, ECE, position-bias report

`mlx-classifiers battery <file>` runs the battery against a runner (fake for
plumbing, real locally), reports accuracy per question type, ECE raw and
calibrated, and the mean probability shift between option orderings.

## Acceptance Criteria
- **Given** the fake runner **when** the battery runs in CI **then** the report
  is produced and the schema round-trip holds for every case.
- **Given** the real model **when** run locally **then** the report is written
  under `bench_runs/` with model id and git revision.
- **Given** `--calibration cfg` **then** calibrated ECE uses that config's
  temperature for the runner's model and each case's readout mode; in every run
  a temperature is also fitted on the battery itself (seeded train / held-out
  split, `fitting.calibrate_records`) and its held-out ECE before and after is
  reported.
- **Given** `--permutations N` (N > 1) **then** each `choice` question is
  re-asked with its options in the `permutation_orders` orderings and the
  report gives the mean over questions of the mean max-prob shift
  (`max_o |p_k(o) - p_0(o)|`, ordering k against the declared order 0) and the
  share of orderings whose answer differs.
- **Given** perturbation groups **then** the report gives the share of
  variants whose answer differs from their base, the share of groups with any
  flip, and a table of the flipped groups with top probabilities.

## Implementation notes
- `mlx-classifiers battery FILE [--runner fake|mlx] [--model M]
  [--permutations N] [--calibration cfg] [--out bench_runs]` writes
  `battery-<stamp>.md` and `.json` (env from `bench.environment`: model, git
  rev, machine, `pmset` power state; summary; every question's raw and
  calibrated rows). Exit status 1 if any case fails the round trip.
- Each case is decided as written; accuracy is the raw argmax (temperature
  does not move it). Raw rows are read from `raw_probabilities` / `raw_noul`,
  calibrated rows are `softmax(log raw / T)`, the same identity `fitting`
  uses. Permuted orderings are built as wire requests with the `criteria` map
  reordered, so the measurement goes through the public request path only.
- The real-model test runs 17 cases (every slice, groups `robot-03` and
  `robot-05`) with permutations 2.

### First full run (2026-10-16, gemma-4-E2B-it-qat-4bit, first-token, permutations 4)

Low power mode was on (`lowpowermode 1`, AC power), Apple M5, rev `e7629db`
plus this story's working tree; 343 questions plus 669 permuted re-asks in
2 min 22 s. No calibration config existed, so the calibrated column is the
battery-fitted temperature on the held-out split.

| type | n | accuracy | mean conf | ECE raw |
|---|---|---|---|---|
| choice | 223 | 0.771 | 0.888 | 0.124 |
| noul | 60 | 0.900 | 0.959 | 0.071 |
| score | 60 | 0.683 | 0.887 | 0.238 |
| all | 343 | 0.778 | 0.900 | 0.122 |

- Battery-fitted T = 1.674 (240 train / 103 held out): held-out ECE 0.127
  before, 0.078 after. The model is over-confident, most on `score`.
- By source: hand-routing 1.000, hand-noul 0.900, agent-router 0.760,
  arc-challenge 0.700, hand-score 0.683, agent-robot 0.593. By difficulty:
  easy 0.893, hard 0.797, near-miss 0.526.
- Position bias: mean max-prob shift 0.142 (max 0.961); the answer changes in
  15.2% of permuted orderings.
- Prefix perturbation: 12.7% of 79 variants answer differently from their
  base; 9 of 30 groups (30%) flip at least once, all of them robot cases. The
  `p2` reword ("for a small wheeled robot") is the main flipper: it moves
  `robot-01`, `-05`, `-09` and `-12` from a confident wrong `forward` to the
  gold answer. The robot block's dominant error is a `forward` prior that
  incidental text can tip either way, which is the probe's failure seen from
  the other side: in this wording `robot-01` already says `forward` (0.88)
  at base, where the probe's wording said `left` (0.60).
