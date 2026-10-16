---
type: epic
status: in-progress
title: Gemma 4 E2B INT8 runner on MLX
epic: 2
owner: maddie
updated: 2026-10-16
---

# Epic 2 — Gemma 4 E2B INT8 runner on MLX

The real `Runner`: load Gemma 4 E2B on MLX with the existing INT8 kernels,
cache the `prefix` KV once, prefill only `state` + questions per request,
slice the hidden state to the readout position before the lm_head, and
measure decisions per second. Scope starts with a recon of where the INT8
kernels and any vendored VLM internals live across Maddie's earlier repos
(story 2.1, done: see the [recon](../research/e2b-int8-and-vlm-recon-2026-10.md)).
The recon's main finding is structural: E2B's cache-only prefill skips the 20
KV-consumer layers, so the readout must run them on the readout row only
(story 2.5) or lose most of the prefill speed advantage.

## Stories
| id | title | status |
|---|---|---|
| 2.1 | Recon — locate the E2B INT8 kernels and any vendored VLM internals | completed |
| 2.2 | MLX runner — load, cached prefix, prefill, readout-position slice | pending |
| 2.3 | INT8 kernel wiring into the runner | pending |
| 2.4 | Decisions-per-second benchmark harness | pending |
| 2.5 | KV-consumer layers run on the readout row only | in-progress |
| 2.6 | INT8 attention router with proper Q-side length padding | pending |

## Related
- ADR-0002 (the runner interface is the port seam)
