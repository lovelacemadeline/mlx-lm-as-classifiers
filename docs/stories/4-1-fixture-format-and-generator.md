---
type: story
status: pending
title: Fixture format and generator
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on story 2.2
---

# Story 4.1 — Fixture format and generator

A fixture is one JSON file: request, model id and revision, tokenizer hash,
compiled token ids, readout position, option token ids, raw logits at the
mask, raw probabilities. A generator produces them from the Python runner.

## Acceptance Criteria
- **Given** the generator **when** run twice on the same machine **then** the
  fixture files are byte-identical.
