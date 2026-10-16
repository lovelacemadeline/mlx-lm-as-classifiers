---
type: story
status: pending
title: Battery format and a seed set of labelled decisions
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only
---

# Story 5.1 — Battery format and a seed set of labelled decisions

A JSONL battery: each line is a wire-schema request plus gold answers. Seed it
with a few hundred cases: a slice of a public text-classification set for
`choice`, yes/no cases for `noul`, an ordinal sentiment or severity set for
`score`, and a hand-written set of agent-routing cases in the style the
library will actually see.

## Acceptance Criteria
- **Given** the battery file **when** every line is parsed **then** each
  validates against the request schema and names a gold answer among its
  declared options.
