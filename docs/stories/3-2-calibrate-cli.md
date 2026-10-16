---
type: story
status: pending
title: calibrate CLI — fit temperature, report ECE
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on story 1.5
---

# Story 3.2 — calibrate CLI — fit temperature, report ECE

`mlx-classifiers calibrate <decisions.jsonl>`: fit a temperature per model and
readout mode from labelled decisions, write it to the model config, print ECE
before and after on a held-out split.

## Acceptance Criteria
- **Given** a labelled decisions file **when** run **then** the fitted
  temperature is persisted and ECE before/after is printed.
