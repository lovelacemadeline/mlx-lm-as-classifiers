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

## Seed material from the first real probe (2026-10-16, E2B qat-4bit, first-token mode)

A four-way robot `choice` (forward / left / right / stop) with prefix "You are a
robot controller. Decide the next action." The model is responsive to the
state but marginal on the hard case, and the marginal case flips with
incidental prompt text, which is exactly what the battery must catch:

| state | choice | top p |
|---|---|---|
| wall 10 cm ahead, doorway to the left | left | 0.60 |
| same, plus "The robot is in a room." in the prefix and "Battery 80%." in the state | forward | 0.83 |
| path ahead clear 5 m, wall to the left | forward | 0.95 |
| robot on fire, battery 2% | stop | 0.97 |
| obstacle ahead, only open direction is right | right | 0.98 |

Position bias was ruled out for the wrong case: permutations=4 and moving
`forward` to the last slot both still chose forward, and sequence mode agreed.
So the battery needs a hand-written block of near-miss agent cases like the
second row, not only obvious ones, and should report the flip rate under small
prefix perturbations.
