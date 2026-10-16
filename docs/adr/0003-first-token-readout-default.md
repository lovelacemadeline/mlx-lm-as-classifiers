---
type: decision
status: active
title: First-token readout over single-token labels is the default; sequence scoring is opt-in
owner: maddie
updated: 2026-10-16
---

# ADR-0003 — First-token readout by default

## Status
Active.

## Context
Two ways to pick between declared options without decoding:
(a) **first-token readout**: bind each option to a single-token label, read the
next-token distribution at the last prefill position, softmax over those token
ids; (b) **sequence scoring**: teacher-force each full option string after the
shared prefix and compare length-normalised loglikelihoods. (b) is one shared
prefill plus one small batched pass per option; (a) is one prefill and a
gather. A prefix-tree variant of (a) handles multi-token options by comparing at
the first distinguishing token.

## Decision
**(a) is the default.** Options are bound to single-token labels (letters, or a
verified single-token word set) at request-compile time; a label that
tokenises to more than one token is a hard error, never a silent fallback.
**(b) is available as `options.readout: "sequence"`** for options whose first
token does not discriminate (long phrases). Both modes return the same answer
shape.

## Consequences
- Cheapest possible decision: one forward pass, one gather, one softmax.
- Position/label bias is real; `options.permutations` averages over orderings.
- Calibration is fitted per model and per readout mode.

## Alternatives rejected
- **Decode with a logit mask then stop**: strictly a superset of the work in
  (a) with an extra decode step. Rejected.
- **Perplexity of each full option as the only mode**: more passes, needs length
  normalisation, no cheaper. Kept as opt-in only.
