---
type: story
status: pending
title: Vision encoder 3x on the 26B — token budget and SDPA width, under the 6.4 tolerances
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; follows 6.6
---

# Story 6.7 — Vision encoder 3x on the 26B

Story 6.6 made the encode bit-exactly faster: E2B 193 → 138 ms, the 26B 659 →
598 ms by profile. It also showed that what remains is matmul (about 327 ms on
the 26B, bf16 at 10 to 11 TFLOP/s on the MLP shapes) plus SDPA at head_dim 72
padded to 80 (252 ms, 3.4 TFLOP/s). INT8 on the tower linears was measured and
gives nothing at these shapes. The 3x on the 26B therefore needs a change that
is not bit-exact. This story chooses one from 6.6's data, in this order:

1. **Soft-token budget 140** (`vision_soft_tokens_per_image` override in the
   runner): 120 soft tokens. The 26B tower drops 590 → 236 ms (2.5x) and E2B
   138 → 59 ms, and each frame prefills about 150 fewer tokens. The gate is
   gridworld battery accuracy at 140 vs 280 on both models, then regenerated
   image fixtures with the budget recorded in them. 70 (63 tokens, 105 ms on
   the 26B) is the next step if 140 holds.
2. **SDPA width 128 on the 26B** (pad 72 → 128): about 110 ms more per frame.
   It is not bit-exact (max |d| 1e-3 per call), so the gate is 6.4's tolerances
   (option-label logits within `LOGIT_TOL`, same argmax), not equality. It
   stacks with 1.
3. Encoding the next frame during the current decision is a latency measure
   only (one GPU); it is out of scope unless the gridworld loop has idle GPU
   time.

## Acceptance Criteria
- **Given** the chosen lever(s) **when** the image bench reruns on the 26B
  (`--int8`, low power off, swap noted) **then** encoder ms per new frame is at
  most a third of 6.6's slow-path profile (659 ms).
- **Given** the chosen lever(s) **then** the gridworld battery accuracy does not
  drop (both models), 6.4 parity passes within its tolerances, and any fixture
  whose numbers change is regenerated with the setting recorded.
