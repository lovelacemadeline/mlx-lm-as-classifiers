---
type: story
status: pending
title: Sequence-scoring readout mode
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; opt-in mode per ADR-0003, built after 1.4 and 1.5
---

# Story 1.6 — Sequence-scoring readout mode

`options.readout: "sequence"`: score each full option string by teacher-forced
loglikelihood after the shared prefix, length-normalised, batched across
options. Same `Answer` shape as first-token mode.

## Acceptance Criteria
- **Given** two multi-token options whose first tokens collide **when**
  readout is `sequence` **then** a probability is returned for each.
- **Given** the fake runner **when** both modes run on single-token labels
  **then** both return the same argmax.
