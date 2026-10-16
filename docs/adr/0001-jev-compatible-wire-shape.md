---
type: decision
status: active
title: Adopt TypeSafe Jev's /v1/systemone wire shape, with x-local extensions
owner: maddie
updated: 2026-10-16
---

# ADR-0001 — Jev-compatible wire shape

## Status
Active.

## Context
TypeSafe's Jev defines a small request/response contract for typed decisions:
one `state`, named `questions` of type `noul` / `choice` / `score`, answers
with `probabilities` and `confidence`. Gateways (LiteLLM, Zuplo) already proxy
it and the open clones mirror it. We need *a* contract and will reuse this code
across several projects (robot controller, judge endpoint, iPhone port).

## Decision
**Match Jev's native field names and question types exactly, and add our
extensions under clearly marked x-local fields** (`prefix`, `options`,
`raw_*`, `expected_score`, `cached_tokens`, `timing_ms`, `debug`). The
JSON Schema in `schema/` is the contract; the Swift port generates its types
from it.

## Consequences
- Zero bikeshedding on names; drop-in behind existing gateways.
- We must define `confidence` ourselves (Jev does not publish its definition);
  see [wire-schema](../reference/wire-schema.md).
- If TypeSafe changes their API we are not obliged to follow; the x-local
  fields are ours and the schema is versioned independently.

## Alternatives rejected
- **OpenAI chat-completions with logprobs**: forces clients to parse tokens and
  gives no typed answers. Rejected.
- **Our own schema from scratch**: no interop benefit, same work. Rejected.
