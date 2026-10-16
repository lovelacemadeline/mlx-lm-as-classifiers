---
type: story
status: pending
title: Delta prefix caching on the longest common token prefix
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; depends on 3.1's PrefixLRU and 2.2's split prefill
---

# Story 3.4 — Delta prefix caching

Today the server caches KV only for the declared `prefix` field. For a
sequential controller most of the state repeats tick to tick (rules, map,
inventory) and only a short tail changes, and a transcript-style memory grows
by appending. Cache on the **longest common token prefix** between the new
request's full token sequence and any cached sequence (mlx-lm's PromptTrie or
an equivalent), trim the cache to that length, and prefill only the delta.
Report the hit length in `usage.cached_tokens` and keep the
`x-prefix-cache` header honest.

Also needed by the gridworld's transcript memory mode.

## Acceptance Criteria
- **Given** two consecutive requests sharing the first N tokens **when** the
  second runs **then** `cached_tokens` is N and only the remainder is prefilled
  (assert via the runner's recorded call shapes).
- **Given** a 1024-token state with a 50-token tail change **when** benchmarked
  **then** decisions/s approaches the 128-token figure.
- **Given** the RotatingKVCache sliding layers **when** a cache is trimmed to a
  prefix length beyond the window **then** results match an uncached prefill
  (argmax and probabilities within the fixture tolerance).
