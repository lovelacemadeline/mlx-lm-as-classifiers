---
type: story
status: pending
title: Memory budget — stop keeping two copies of INT8-patched weights, then quantise the KV store
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only
---

# Story 2.9 — Memory budget

The 26B with everything on sits well above its 15 GB of 4-bit weights, and
with a viewer and an agent sharing the machine the box swaps. Measured
pieces so far: the INT8 wrappers keep the stock linear beside the int8 copy
for the under-64-row fallback (E2B went 3.4 to 5.2 GB resident in story 2.3;
the 26B expert path added about 1.4 GB in 2.8); the vendored vision tower;
the delta cache store (2 GiB cap by default, story 3.4); a separate planner
cache; and, when `--plan-model` is used, a second model entirely. Maddie's
earlier gemma4-turboquant-mlx fork ran four 128k streams of the 26B on this
machine with PolarQuant 2-bit KV, so the KV side has a known answer; the
weight duplication does not.

In order:
1. **Audit**: a `mlx-classifiers memory` report that loads a model with each
   flag and prints resident and MLX active/peak memory per component.
2. **Drop the duplicate weights**: make the INT8 fallback path run on the
   int8 weights (pad rows to the kernel's minimum, or dequantise on the fly
   for small rows) so the stock linears can be freed after patching.
3. **Quantise the KV store**: vendor the PolarQuant 2-bit codec from
   gemma4-turboquant-mlx for stored (not live) cache entries, so the delta
   store holds many more sequences in the same budget; measure the effect
   on probabilities against the fixtures.
4. Size the delta store cap by model and expose it on the CLI and server.

## Acceptance Criteria
- **Given** the audit **when** run for E2B and the 26B with int8 and vision
  **then** each component's bytes are listed.
- **Given** step 2 **when** the 26B loads with `int8=True` **then** resident
  memory is within 10% of the stock load and the parity gate still passes.
- **Given** step 3 **when** the delta bench runs with a 20-step transcript
  **then** store bytes drop by at least 4x with argmax unchanged on fixtures.
