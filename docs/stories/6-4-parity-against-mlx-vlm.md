---
type: story
status: pending
title: Parity test against mlx-vlm as a dev-only reference
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; Epic 6 not started
---

# Story 6.4 — Parity test against mlx-vlm as a dev-only reference

A `vision-dev` extra installs mlx-vlm; a test encodes the same image through both paths and asserts the merged embeddings match within tolerance. Skipped when the extra is absent.

## Acceptance Criteria
- **Given** mlx-vlm installed **when** the parity test runs **then** max abs embedding difference is below the stated tolerance.
