---
type: story
status: pending
title: Decisions-per-second benchmark harness
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on 2.2
---

# Story 2.4 — Decisions-per-second benchmark harness

A benchmark that reports decisions per second against state length, with and
without the prefix cache, INT8 on and off, and the prefill tok/s it implies.

## Acceptance Criteria
- **Given** a state-length sweep **when** the harness runs **then** it writes a
  table of decisions/s and prefill tok/s with the machine and power mode
  recorded.
