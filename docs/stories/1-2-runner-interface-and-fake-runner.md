---
type: story
status: ready-for-dev
title: Runner interface and deterministic fake runner
owner: maddie
updated: 2026-10-16
anchor: none  # not yet built; anchors added when src/mlx_classifiers/runner.py lands
---

# Story 1.2 — Runner interface and deterministic fake runner

The seam the Swift port swaps (ADR-0002). A `Runner` protocol with:
`tokenize(text) -> list[int]`, `prefill(tokens, prefix_cache=None) -> Readout`
where `Readout` carries logits (or hidden states plus an lm_head callable) at
the requested positions only, and `cache_prefix(tokens) -> PrefixCache`.
Plus `FakeRunner`: a deterministic runner over a toy tokenizer whose logits are
a fixed function of the token sequence, so every Epic 1 story is tested
without weights.

## Acceptance Criteria
- **Given** the same token sequence **when** `FakeRunner.prefill` runs twice
  **then** the readout logits are identical.
- **Given** a prefix cache **when** `prefill` is called with it **then** only
  the suffix tokens are counted in `usage.input_tokens` and `cached_tokens`
  equals the prefix length.
- **Given** a readout position list **when** `prefill` returns **then** logits
  exist only for those positions (no full-sequence vocab projection).
