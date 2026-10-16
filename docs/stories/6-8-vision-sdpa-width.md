---
type: story
status: pending
title: Vision encoder SDPA width 128 on the 26B, under the 6.4 tolerances
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; follows 6.7
---

# Story 6.8 — Vision encoder SDPA width on the 26B

Story 6.7 kept the soft-token budget at 280: smaller budgets cost the 26B
battery accuracy. At 280 the 26B tower is still about 590 ms per frame, 252 ms
of it SDPA at head_dim 72 padded to 80 (3.4 TFLOP/s). Story 6.6 measured
`[1, 16, 2430, d]` SDPA at d = 128 in 5.2 ms against 9.35 ms at 80. That is
about 110 ms per frame (598 → about 490 ms). It is not bit-exact (max |d| 1e-3
per call).

Constraints from 6.7: the 26B's compiled fast path is bit-equal to the
vendored path only on the pinned mlx 0.32.0 (max |d| 5.0 on 0.32.4). Measure
any width change on the pin, and re-run 6.6's GPU bit-equality before an mlx
upgrade.

## Acceptance Criteria
- **Given** the 26B tower with SDPA padded to 128 **when** the 6.4 parity test
  and the image fixtures run **then** option logits stay within `LOGIT_TOL`
  with the same argmax, and the gridworld battery accuracy at 280 does not
  drop.
- **Given** the change **when** profiled (low power off, swap noted) **then**
  encoder ms per frame on the 26B is recorded against 6.6's 598 ms.
