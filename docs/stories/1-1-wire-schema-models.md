---
type: story
status: ready-for-dev
title: Wire schema models and validation
owner: maddie
updated: 2026-10-16
anchor: none  # not yet built; anchors added when src/mlx_classifiers/schema.py lands
---

# Story 1.1 — Wire schema models and validation

Pydantic v2 models mirroring `schema/systemone-request.schema.json` and
`schema/systemone-response.schema.json`: `NoulQuestion`, `ChoiceQuestion`,
`ScoreQuestion`, the `Request`, and the three `Answer` types. The JSON Schema
files are the contract; the models must round-trip them.

## Acceptance Criteria
- **Given** the example request in [wire-schema](../reference/wire-schema.md)
  **when** parsed **then** it validates and re-serialises to equal JSON.
- **Given** a `choice` with fewer than two criteria, or a `score` with fewer
  than two levels **when** parsed **then** validation fails with the field named.
- **Given** the pydantic models **when** exported with `model_json_schema()`
  **then** a test asserts every required field and enum in `schema/*.json` is
  present (drift tripwire between the two).
- **Given** an unknown top-level field **when** parsed **then** it is rejected
  (`additionalProperties: false` mirrored).
