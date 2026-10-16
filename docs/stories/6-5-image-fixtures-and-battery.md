---
type: story
status: pending
title: Image fixtures and a battery slice with images
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; Epic 6 not started
---

# Story 6.5 — Image fixtures and a battery slice with images

Fixtures pin the image sha256 beside the token ids and probabilities; the battery gains an image slice (gridworld frames with gold actions from the oracle) so text and image modes are compared on identical states.

## Acceptance Criteria
- **Given** the image fixtures **when** regenerated **then** they are byte-identical; **given** the image battery slice **when** run **then** accuracy is reported beside the text slice.
