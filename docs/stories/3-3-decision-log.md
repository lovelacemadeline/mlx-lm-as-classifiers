---
type: story
status: pending
title: Decision log — every answer with its probabilities, for later refits
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on 3.1
---

# Story 3.3 — Decision log — every answer with its probabilities, for later refits

Append every served decision (request hash, model, raw and calibrated
probabilities, timing) to a JSONL log so calibration can be refit when states
drift from the fitting set.

## Acceptance Criteria
- **Given** logging on **when** a decision is served **then** one JSONL line is
  appended in the format the calibrate CLI accepts once a label is added.
