---
type: epic
status: completed
title: Golden fixtures for the Swift + vision port
epic: 4
owner: maddie
updated: 2026-10-17
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
| 4.2 | Fixture checker usable from another repo | completed |
| 4.3 | Per-fixture gate — near-tied gridworld cases on answer and ids, clear-winner image fixtures at every budget | completed |

## Related
- ADR-0002
- [Fixture format](../reference/fixture-format.md)
