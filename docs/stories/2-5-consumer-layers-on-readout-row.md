---
type: story
status: ready-for-dev
title: KV-consumer layers run on the readout row only
owner: maddie
updated: 2026-10-16
anchor: none  # not yet built; the model-structural win identified by the recon (§D)
---

# Story 2.5 — KV-consumer layers run on the readout row only

E2B shares K/V from layers 0 to 14 into layers 15 to 34. Cache-only prefill
runs only the producers, which is where the fast prefill number comes from, but
a logit readout needs the consumers. Run producers over all tokens, then run
consumers over the readout row(s) only, slicing the residual and
`per_layer_inputs` to those rows. Exact, because consumers read shared K/V plus
their own residual. See the
[recon](../research/e2b-int8-and-vlm-recon-2026-10.md) §D.

## Acceptance Criteria
- **Given** a prompt **when** the sliced path and the full forward both run
  **then** readout logits match to within bf16 noise on every fixture.
- **Given** a 2k-token prompt **when** profiled **then** consumer layers see
  only the readout rows (test by asserting the hidden-state shape entering
  layer 15).
- **Given** the sliced path **when** benchmarked (story 2.4) **then** decisions/s
  approaches cache-only prefill tok/s divided by prompt length, not a third
  of it.
