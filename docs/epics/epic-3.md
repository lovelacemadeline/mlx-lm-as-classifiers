---
type: epic
status: in-progress
title: Serving and calibration
epic: 3
owner: maddie
updated: 2026-10-16
---

# Epic 3 — Serving and calibration

A `POST /v1/systemone` route served beside the existing chat endpoint from the
same loaded model, and a `calibrate` CLI that fits the temperature from a
labelled decisions file and reports held-out ECE. This is what makes the Qwen
27B judge endpoint and the Gemma classifier the same deployable.

## Stories
| id | title | status |
|---|---|---|
| 3.1 | /v1/systemone route beside chat completions | completed |
| 3.2 | calibrate CLI — fit temperature, report ECE | completed |
| 3.3 | Decision log — every answer with its probabilities, for later refits | completed |
| 3.4 | Delta prefix caching on the longest common token prefix | pending |

## Related
- [wire-schema](../reference/wire-schema.md)
- [serving](../reference/serving.md)
