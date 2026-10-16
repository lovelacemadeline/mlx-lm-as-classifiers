---
type: story
status: pending
title: Image blocks in the wire state and in prompt compilation
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; Epic 6 not started
---

# Story 6.1 — Image blocks in the wire state and in prompt compilation

Extend the request schema so a list-valued `state` may contain `{"type": "image/png", "base64": ...}` blocks beside strings (additive, x-local, v1 stays). Prompt compilation places each image at the chat template's image placeholder with the right number of soft-token slots, keeps text order, and records the image hashes for fixtures. Text-only requests compile byte-identically to today (drift test).

## Acceptance Criteria
- **Given** a state with one text and one image block **when** compiled **then** the token sequence contains the placeholder run at the image position and the readout position is unchanged in meaning.
