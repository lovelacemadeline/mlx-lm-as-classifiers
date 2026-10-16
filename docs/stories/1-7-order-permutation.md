---
type: story
status: pending
title: Order permutation for position-bias reduction
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; built after 1.3 and 1.5
---

# Story 1.7 — Order permutation for position-bias reduction

`options.permutations: N` compiles each question N times with option order
permuted (deterministic seed), averages probabilities, and reports the spread.

## Acceptance Criteria
- **Given** `permutations: 2` **when** a choice runs **then** two prompts are
  compiled, labels stay bound to options across orderings, and the answer is
  the mean of the two distributions.
