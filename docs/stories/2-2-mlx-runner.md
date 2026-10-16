---
type: story
status: pending
title: MLX runner — load, cached prefix, prefill, readout-position slice
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; pins decided: mlx==0.32.0, mlx-lm==0.31.3 (recon doc §E)
---

# Story 2.2 — MLX runner — load, cached prefix, prefill, readout-position slice

Implement `Runner` (story 1.2) on mlx-lm for Gemma 4 E2B: load, build and
reuse a prefix KV cache, prefill the suffix, and slice the final hidden state
to the readout positions **before** the lm_head. Depends on stock PyPI mlx-lm 0.31.3 and mlx 0.32.0 per the [recon](../research/e2b-int8-and-vlm-recon-2026-10.md); no fork. Story 2.5 owns the KV-consumer trick; this story may run the full forward first and must leave the seam for 2.5 (Gemma vocab is large; the full
projection is the dominant cost at long prompts).

## Acceptance Criteria
- **Given** a request with a cached prefix **when** run **then** only suffix
  tokens are prefilled and `cached_tokens` reports the prefix length.
- **Given** a 2k-token prompt **when** profiled **then** no vocab-sized
  projection runs over positions other than the readout positions.
- **Given** the same request **when** run twice **then** raw probabilities are
  identical.
