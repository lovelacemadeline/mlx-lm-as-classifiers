---
type: epic
status: in-progress
title: Golden fixtures for the Swift + vision port
epic: 4
owner: maddie
updated: 2026-10-16
---

# Epic 4 — Golden fixtures

The conformance contract for the second repo (ADR-0002): a fixture format
(state, questions, model id, expected token ids at the readout position,
expected raw probabilities), a generator that produces them from the Python
runner, and a checker that any port runs against them.

## Stories
| id | title | status |
|---|---|---|
| 4.1 | Fixture format and generator | completed |
| 4.2 | Fixture checker usable from another repo | pending |

## Related
- ADR-0002
- [Fixture format](../reference/fixture-format.md)
