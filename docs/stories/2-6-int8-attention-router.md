---
type: story
status: pending
title: INT8 attention router with proper Q-side length padding
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on 2.3; the router lives in int8-mlx agent_e4b_wire4.py
---

# Story 2.6 — INT8 attention router with proper Q-side length padding

Bring the TP-EVERYTHING attention router across behind a flag, and fix the
router engagement condition (prompt length a multiple of 64) with Q-side
padding and masking, never filler tokens. Batch-1, causal-only constraints are
acceptable for the classifier path and must be asserted, not assumed.

## Acceptance Criteria
- **Given** any prompt length **when** the router is on **then** it engages
  (no silent stock fallback) and fixtures still match within tolerance.
