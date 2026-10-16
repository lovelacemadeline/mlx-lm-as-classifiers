---
type: story
status: pending
title: Bounded-context mode — window the global layers with the rules prefix pinned as sinks
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on 3.4 delta caching and the gridworld transcript mode to measure quality
---

# Story 2.7 — Bounded-context mode

E2B already runs 30 of 35 layers on a 512-token rotating window. Give the 5
global layers a rotating cache too (window W, e.g. 4k), with
`keep = len(prefix)` so the rules prefix stays pinned as attention sinks
(StreamingLLM) while the transcript slides behind it. Then per-tick cost is
bounded by W regardless of episode length, the cache never goes cold, and
snapshots at branch points are bounded by W as well.

Known constraints: a rotating cache cannot rewind to an earlier prefix once
content has slid out, so longest-common-prefix hits are forward-only
(append-only transcript is fine; branching needs a snapshot). The model sees a
position gap between the pinned block and the window; quality past the trained
context is the open question.

## Acceptance Criteria
- **Given** a 20k-token append-only transcript **when** ticks run in this mode
  **then** per-tick prefill cost and resident KV are flat after the first W.
- **Given** the battery and a gridworld transcript run **when** global layers
  are windowed vs not **then** accuracy, agreement and oscillation are reported
  side by side at W in {1k, 4k, 8k}.
- **Given** a branch (permutations) **when** taken **then** the snapshot size is
  bounded by W and results match the unbranched run.
