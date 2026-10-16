---
type: story
status: completed
title: calibrate CLI — fit temperature, report ECE
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/fitting.py::calibrate_records"
  - symbol: "src/mlx_classifiers/fitting.py::CalibrationConfig"
  - symbol: "src/mlx_classifiers/fitting.py::main"
  - test: "tests/test_fitting.py::test_fit_recovers_the_overconfidence_and_lowers_heldout_ece"
  - test: "tests/test_fitting.py::test_calibrate_cli_prints_ece_and_merges_config"
  - test: "tests/test_fitting.py::test_server_applies_the_fitted_temperature"
---

# Story 3.2 — calibrate CLI — fit temperature, report ECE

`mlx-classifiers calibrate <decisions.jsonl>`: fit a temperature per model and
readout mode from labelled decisions, write it to the calibration config,
print ECE before and after on a held-out split.

## Acceptance Criteria
- **Given** a labelled decisions file **when** run **then** the fitted
  temperature is persisted (merged into the config, other entries kept) and
  ECE and NLL before and after are printed for the held-out split.
- **Given** synthetic decisions whose labels are drawn at a known temperature
  above 1 (an over-confident model) **when** fitted **then** the recovered
  temperature is close to it and held-out ECE and NLL both fall.
- **Given** a fitted config **when** the server loads it **then** decisions
  are served at that temperature, and at `T = 1` with `calibrated: false`.

## Implementation notes
- `src/mlx_classifiers/fitting.py`: `calibrate_records` groups labelled
  records by `(model, readout)`, splits with a seeded shuffle, fits
  `TemperatureScaler` (story 1.5) on `log(raw_probabilities)` of the train
  split, and measures on the held-out split. `CalibrationConfig` is the JSON
  file; `serve --calibration` loads it into `create_app(calibration=...)`.
- Config location, file shape and the input record fields are documented in
  [serving](../reference/serving.md#calibration-config).
- Tested with FakeRunner decisions labelled by sampling at `T = 3`.
