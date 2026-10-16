---
type: epic
status: ready-for-dev
title: Schema and readout core (model-agnostic)
epic: 1
owner: maddie
updated: 2026-10-16
---

# Epic 1 — Schema and readout core

Everything that does not need weights: the pydantic models for the wire schema
(ADR-0001), compiling a request into a prompt with a known readout position,
binding options to single-token labels, the prefix-tree readout, masked
softmax, temperature calibration, and a deterministic fake runner so all of it
is unit-tested on any machine. The real runner (Epic 2) plugs in behind the
`Runner` interface defined here (ADR-0002).

## Stories
| id | title | status |
|---|---|---|
| 1.1 | Wire schema models and validation | ready-for-dev |
| 1.2 | Runner interface and deterministic fake runner | ready-for-dev |
| 1.3 | Prompt compilation — chat template, prefix/state split, readout position | ready-for-dev |
| 1.4 | Option binding and prefix-tree first-token readout | ready-for-dev |
| 1.5 | Masked softmax, temperature calibration, confidence | ready-for-dev |
| 1.6 | Sequence-scoring readout mode | pending |
| 1.7 | Order permutation for position-bias reduction | pending |

## Related
- [PRD](../product/prd-system-one-classifier.md), [wire-schema](../reference/wire-schema.md)
- ADR-0001, ADR-0002, ADR-0003
