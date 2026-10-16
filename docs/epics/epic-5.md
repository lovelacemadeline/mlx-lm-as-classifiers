---
type: epic
status: pending
title: Static decision battery (accuracy, calibration, position bias)
epic: 5
owner: maddie
updated: 2026-10-16
---

# Epic 5 — Static decision battery

A labelled set of decisions across the three question types, run through the
library to report accuracy, ECE before and after calibration, and
position-bias spread. It is this implementation's regression test, so it lives
here. The sequential live-loop benchmark (gridworld, APM, oracle agreement)
is deliberately a separate repo that consumes the wire schema, so it can
benchmark any backend.

## Stories
| id | title | status |
|---|---|---|
| 5.1 | Battery format and a seed set of labelled decisions | pending |
| 5.2 | Battery runner — accuracy, ECE, position-bias report | pending |

## Related
- [PRD](../product/prd-system-one-classifier.md), ADR-0001
