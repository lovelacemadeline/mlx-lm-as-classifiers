---
type: story
status: completed
title: Recon — locate the E2B INT8 kernels and any vendored VLM internals
owner: maddie
updated: 2026-10-16
anchor: none  # read-only investigation; its deliverable is docs/research/e2b-int8-and-vlm-recon-2026-10.md
---

# Story 2.1 — Recon — locate the E2B INT8 kernels and any vendored VLM internals

Before deciding what Epic 2 vendors or depends on: find where the Gemma E2B
INT8 kernels live, how they are wired into mlx-lm, which fork or tag they need,
the provenance of the ~9k tok/s prefill figure, whether any sibling repo
already vendored Gemma VLM internals (vision tower, image tokens) without
depending on mlx-vlm, and any E2B-specific prefill behaviour (per-layer
embeddings, KV sharing, layer skip) that a prefill-only classifier must respect.
Candidates: `july/int8-mlx`, `aug/poorly-ported-gemma4-int8-optimisations`,
`may/gemma4-turboquant-mlx`, `july/e4b-vs-e2b`, `apr/gemma4`.

## Acceptance Criteria
- **Given** the sibling repos **when** the recon completes **then** a research
  doc under `docs/research/` records, per question, the repo, file, and git
  state, and flags anything stale or contradictory.
- **Given** the research doc **when** read **then** stories 2.2 and 2.3 can
  name their dependency (pin or vendored files) without further digging.

## Result (2026-10-16)
Done; see [the recon research doc](../research/e2b-int8-and-vlm-recon-2026-10.md).
Headlines: kernels monkeypatch stock mlx-lm 0.31.3 (no fork); no VLM internals
were ever vendored; the 9k tok/s figure is a cache-only prefill that skips the
20 KV-consumer layers, which a logit readout must run (now story 2.5).
