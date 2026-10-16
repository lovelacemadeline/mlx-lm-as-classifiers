---
type: story
status: pending
title: Vision encoder speed — 675 ms per frame on the 26B is not the hardware
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; follows 6.3
---

# Story 6.6 — Vision encoder speed

Story 6.3's bench (low power off) measured the encode of one 160x192
gridworld frame at 198 ms on E2B and 675 ms on the 26B. With delta caching
making frame history free, per-tick cost in image mode is roughly encoder
plus ~395 prefilled tokens, so the encoder is now the largest single cost in
the vision path. Those numbers are far above what a tower of this size
should take on an M5 at full power, which points at the vendored path rather
than the model: dtype wider than needed, no `mx.compile` on the tower or
projector, Python-side work in preprocessing, or kernel recompiles because
shapes are not pinned.

Profile first (preprocess, tower, projector, merge, each behind
`mx.synchronize`), then fix the largest, in order of likelihood above.
Parity gate: the image fixtures and the mlx-vlm parity test (6.4) must still
pass within their tolerances.

## Acceptance Criteria
- **Given** the profile **when** recorded **then** each stage has a measured
  share on both E2B and the 26B.
- **Given** the fixes **when** the image bench reruns **then** encoder ms per
  new frame drops by at least 3x on the 26B with image fixtures unchanged.
