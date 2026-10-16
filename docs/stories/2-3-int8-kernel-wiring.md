---
type: story
status: pending
title: INT8 kernel wiring into the runner
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on 2.1 and 2.2
---

# Story 2.3 — INT8 kernel wiring into the runner

Vendor `w8a8_wiring.py` from `aug/poorly-ported-gemma4-int8-optimisations`
(MLP + projection W8A8, self-contained) into `src/mlx_classifiers/int8/` and
wire it behind a flag, with a parity test against the stock path. The
attention router (`agent_e4b_wire4.py`) is a follow-up (story 2.6) because of
its length-multiple-of-64 and batch-1 constraints. See the
[recon](../research/e2b-int8-and-vlm-recon-2026-10.md) §A, §C.

## Acceptance Criteria
- **Given** INT8 on and off **when** the same fixtures run **then** argmax
  agrees on every case and probabilities agree within a stated tolerance.
