---
type: story
status: pending
title: MoE expert INT8 for the 26B — vendor the expert-GEMM path from the August repo
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; follows 2.3
---

# Story 2.8 — MoE expert INT8

`--int8` gives the 26B MoE about 5% (`bench_runs/bench-20261016-182238`
vs `-182057`): the vendored W8A8 wiring patches dense MLPs and projections
and leaves the expert layers (Router + Experts / SwitchGLU in mlx-lm 0.31.3)
untouched. Maddie's August repo
`~/Documents/dev/2026/aug/poorly-ported-gemma4-int8-optimisations` is the
"Gemma4 int8 MoE prefill harness" and reached roughly 1k tok/s prefill on
the 26B (plugged in). Find its expert-GEMM path (sorted gather, per-expert
W8A8), vendor it with provenance beside `int8/wiring.py`, patch the expert
modules behind the same `int8` flag, and parity-test against the fixtures.

Since [3.6](3-6-delta-cache-fixed-overhead.md), `MLXRunner.load` swaps the
26B's `SwitchGLU` to `moe.SegmentedSwitchGLU` (bit-identical per-expert NAX
tiles for 64 to 256-token chunks; about 300 ms of a 547 ms delta decision is
still expert matmuls, bound by the weight loader). An INT8 expert path has to
replace or compose with that class.

## Acceptance Criteria
- **Given** the 26B with `int8=True` **when** the fixtures run **then** argmax
  agrees and probabilities are within the stated tolerance.
- **Given** the bench at 512 and 1024 tokens, uncached **when** INT8 is on
  **then** prefill tok/s improves materially over stock, recorded with power
  state.
