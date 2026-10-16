---
type: story
status: pending
title: Delta cache fixed overhead — the 600 ms floor on the 26B
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; follows 3.4
---

# Story 3.6 — Delta cache fixed overhead

Bench on 2026-10-16 (low power mode, `bench_runs/bench-20261016-182057`):
with delta caching the 26B MoE prefills only 102 tokens per decision at every
state length, yet p50 stays at 670 to 720 ms, so decisions/s plateaus at
about 1.5. E2B's equivalent floor is about 30 ms. The remaining cost is not
prefill (prefill tok/s reads 140 to 170 in those rows because the clock
includes the overhead). Suspects, in order: deep-copying the stored KV entry
on lookup and again on store (the 26B's KV per token is far larger than
E2B's), the L=1 readout pass through the MoE (expert gather), and the store's
byte accounting. Profile each with `mx.synchronize` fences, then fix the
largest: copy-on-write or trimming in place for the lookup copy; storing only
when the sequence is new; snapshotting lazily.

## Acceptance Criteria
- **Given** the profile **when** recorded in this story **then** each suspect
  has a measured share of the 26B's per-decision wall time.
- **Given** the fix **when** the delta bench reruns on the 26B at 1024 tokens
  **then** p50 drops by at least half with identical probabilities.
