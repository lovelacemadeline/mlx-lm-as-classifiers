---
type: story
status: completed
title: Masked softmax, temperature calibration, confidence
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/calibrate.py::TemperatureScaler"
  - symbol: "src/mlx_classifiers/calibrate.py::softmax"
  - symbol: "src/mlx_classifiers/calibrate.py::expected_calibration_error"
  - symbol: "src/mlx_classifiers/answers.py::build_answer"
  - symbol: "src/mlx_classifiers/decide.py::decide"
  - test: "tests/test_calibrate.py::test_softmax_is_masked_to_declared_options"
  - test: "tests/test_calibrate.py::test_temperature_changes_calibrated_not_raw"
  - test: "tests/test_calibrate.py::test_temperature_scaler_fixes_overconfidence"
  - test: "tests/test_calibrate.py::test_score_is_argmax_and_expected_score_is_weighted_mean"
  - test: "tests/test_calibrate.py::test_decide_one_pass_per_question_no_decode"
---

# Story 1.5 — Masked softmax, temperature calibration, confidence

From readout logits and the option mask: gather, softmax over the masked ids
only, apply an optional fitted temperature, and build the typed `Answer`
(`noul`, `choice`, `score`, with `raw_*`, `confidence`, `expected_score`).
A `TemperatureScaler` that fits one scalar by NLL on labelled decisions and
reports ECE before and after.

## Acceptance Criteria
- **Given** logits over the full vocab **when** readout runs **then**
  probabilities sum to one over the declared options and nothing else appears.
- **Given** a fitted temperature **when** applied **then** `raw_probabilities`
  are untouched and `probabilities` change; `confidence` equals the calibrated
  probability of the returned choice.
- **Given** a synthetic over-confident logit set with known labels **when**
  `TemperatureScaler.fit` runs **then** ECE after is lower than before and the
  temperature is greater than one.
- **Given** a `score` answer **when** built **then** `score` is the argmax
  index and `expected_score` is the probability-weighted mean index.

## Implementation notes
- Pure Python (`math.fsum`, no numpy): the masked row is a handful of floats
  and the Swift port copies these few lines. Ties resolve to the first
  declared option.
- `TemperatureScaler.fit` minimises mean NLL over `log T` by golden-section
  search (NLL is convex in `1/T`, so the optimum is global and the fit is
  deterministic) and returns a `CalibrationReport` with NLL and top-label ECE
  (15 equal-width bins) before and after. On a synthetic set made three times
  too sharp it recovers `T` near three.
- `decide(request, runner, *, temperature=None)` (`decide.py`, re-exported
  from the package) strings compile, bind, prefill and answer together, so
  Epic 2/3 only supply a Runner. The temperature applies when
  `options.calibrated` is true; `debug` fills `readout_position`,
  `option_token_ids`, `raw_logits` (keyed by option) and `temperature`.
- Usage accounting in `decide`: the prefix is cached once per call;
  `input_tokens` sums the suffix tokens each prefill computed and
  `cached_tokens` sums the prefix tokens each one reused (so three questions
  report three times the prefix length). The one-off `cache_prefix` cost is
  not counted.
