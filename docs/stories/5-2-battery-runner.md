---
type: story
status: pending
title: Battery runner — accuracy, ECE, position-bias report
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on Epic 1 and 5.1
---

# Story 5.2 — Battery runner — accuracy, ECE, position-bias report

`mlx-classifiers battery <file>` runs the battery against a runner (fake for
plumbing, real locally), reports accuracy per question type, ECE raw and
calibrated, and the mean probability shift between option orderings.

## Acceptance Criteria
- **Given** the fake runner **when** the battery runs in CI **then** the report
  is produced and the schema round-trip holds for every case.
- **Given** the real model **when** run locally **then** the report is written
  under `bench_runs/` with model id and git revision.
