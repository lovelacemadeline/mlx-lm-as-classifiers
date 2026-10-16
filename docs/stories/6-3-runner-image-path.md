---
type: story
status: pending
title: MLXRunner image path — encode, merge embeddings, same split prefill
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; Epic 6 not started
---

# Story 6.3 — MLXRunner image path — encode, merge embeddings, same split prefill

`MLXRunner.prefill` accepts input embeddings: text embeddings with image soft tokens merged in, then the existing cache-only / full split and per-layer inputs. INT8 patches apply unchanged (they wrap text linears).

## Acceptance Criteria
- **Given** an image request **when** run twice **then** raw probabilities are identical; **given** INT8 on and off **then** argmax agrees within the fixture tolerance.
