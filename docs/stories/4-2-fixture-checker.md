---
type: story
status: pending
title: Fixture checker usable from another repo
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on 4.1
---

# Story 4.2 — Fixture checker usable from another repo

A checker that takes a fixture directory and a callable (or a results JSON
produced by a port) and reports per-case token-id equality and probability
distance, so the Swift repo can run it in CI.

## Acceptance Criteria
- **Given** a results file with one wrong token id **when** checked **then** the
  case fails naming the position and both ids.
