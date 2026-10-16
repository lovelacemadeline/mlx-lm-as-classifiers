---
type: story
status: completed
title: Decision log — every answer with its probabilities, for later refits
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/decision_log.py::DecisionLog"
  - symbol: "src/mlx_classifiers/decision_log.py::decision_records"
  - symbol: "src/mlx_classifiers/decision_log.py::request_hash"
  - test: "tests/test_decision_log.py::test_each_served_question_appends_one_record"
  - test: "tests/test_decision_log.py::test_round_trip_serve_log_label_calibrate"
---

# Story 3.3 — Decision log — every answer with its probabilities, for later refits

Append every served decision (request hash, model, readout mode, raw and
calibrated probabilities, timing) to a JSONL log so calibration can be refit
when states drift from the fitting set.

## Acceptance Criteria
- **Given** logging on **when** a decision is served **then** one JSONL line
  per answered question is appended, in the format the calibrate CLI accepts
  once a label is added; a rejected (422) request appends nothing.
- **Given** a served-then-labelled log **when** `calibrate` runs on it and the
  server restarts with the fitted config **then** decisions are served (and
  logged) at the fitted temperature.

## Implementation notes
- `src/mlx_classifiers/decision_log.py`: `decision_records` turns a request
  and its response into per-question records (options in declared order,
  probability lists aligned to them); `DecisionLog` appends under a lock.
  `serve --log PATH` / `create_app(log=...)`; the server logs on the model
  worker after `decide`.
- One line per answered question rather than per request, so a `label` is a
  single field and each line is one calibration row. The format is
  documented in [serving](../reference/serving.md#decision-log).
