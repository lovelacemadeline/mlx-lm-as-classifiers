---
type: story
status: pending
title: /v1/systemone route beside chat completions
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on Epic 1 and story 2.2
---

# Story 3.1 — /v1/systemone route beside chat completions

A `POST /v1/systemone` route on the same server process as chat completions,
sharing the loaded model and the prefix cache. Returns the wire schema
response; validation errors return 422 with field paths.

## Acceptance Criteria
- **Given** a valid request **when** posted **then** the response validates
  against `schema/systemone-response.schema.json` and `output_tokens` is 0.
- **Given** a chat completion and a decision with the same system text
  **when** both run **then** the second reports `cached_tokens` > 0.
