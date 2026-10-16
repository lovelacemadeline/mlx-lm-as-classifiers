---
type: story
status: ready-for-dev
title: Masked softmax, temperature calibration, confidence
owner: maddie
updated: 2026-10-16
anchor: none  # not yet built; anchors added when src/mlx_classifiers/calibrate.py lands
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
