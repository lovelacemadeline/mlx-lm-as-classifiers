---
type: story
status: completed
title: Sequence-scoring readout mode
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/sequence.py::score_sequences"
  - symbol: "src/mlx_classifiers/calibrate.py::token_logprob"
  - test: "tests/test_sequence.py::test_colliding_multi_token_options_get_probabilities"
  - test: "tests/test_sequence.py::test_sequence_scores_are_length_normalised_teacher_forced_loglik"
  - test: "tests/test_sequence.py::test_both_modes_agree_on_single_token_labels"
  - test: "tests/test_sequence.py::test_both_modes_return_the_same_argmax_on_a_peaked_fake"
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

## Implementation notes
- **What is scored.** In sequence mode the user turn bullets the options and
  asks for one "exactly as written"; each option's text (the `choice`
  criteria key, the `score` level, or Yes/No for `noul`) is resolved in
  context like a first-token label (same leading-space and merge rules) and
  may be any number of tokens. Identically tokenized options are a
  `LabelTokenizationError`.
- **Passes.** The question prompt minus its last token is cached once
  (`cache_prefix`); each option is one `prefill` of
  `[last prompt token] + option tokens` with readout positions at exactly the
  tokens that predict the option. "Batched across options" is realised as one
  small pass per option through the unchanged Runner protocol, not a batch
  dimension: a batched prefill is a runner optimisation for Epic 2 and needs no
  seam change.
- **Score.** Mean log-probability per option token (full-vocab log-softmax via
  `token_logprob`, which uses a row's `logsumexp()` when it has one). The
  scores feed the same `shape_answer` as first-token logits, so temperature,
  `raw_*` and `confidence` behave identically; `debug.raw_logits` holds the
  normalised scores and `debug.option_token_ids` the full option ids.
- **Mode agreement.** With context-free fake logits and option names equal
  to the letter labels, both modes score the same token at the same position
  and agree to 1e-9; with a dominant bias on a pseudo-random fake they return
  the same argmax.
- The request `prefix` cache is not used in this mode (each question caches its
  own prompt); calibration is per readout mode (ADR-0003).
