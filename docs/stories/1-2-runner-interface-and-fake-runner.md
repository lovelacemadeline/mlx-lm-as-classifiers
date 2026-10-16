---
type: story
status: completed
title: Runner interface and deterministic fake runner
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/runner.py::Runner"
  - symbol: "src/mlx_classifiers/runner.py::FakeRunner"
  - symbol: "src/mlx_classifiers/runner.py::Readout"
  - symbol: "src/mlx_classifiers/runner.py::PrefixCache"
  - test: "tests/test_runner.py::test_prefill_is_deterministic"
  - test: "tests/test_runner.py::test_prefix_cache_counts_only_suffix_tokens"
  - test: "tests/test_runner.py::test_logits_only_at_readout_positions"
---

# Story 1.2 — Runner interface and deterministic fake runner

The seam the Swift port swaps (ADR-0002). A `Runner` protocol with:
`tokenize(text) -> list[int]`, `prefill(tokens, prefix_cache=None) -> Readout`
where `Readout` carries logits (or hidden states plus an lm_head callable) at
the requested positions only, and `cache_prefix(tokens) -> PrefixCache`.
Plus `FakeRunner`: a deterministic runner over a toy tokenizer whose logits are
a fixed function of the token sequence, so every Epic 1 story is tested
without weights.

## Acceptance Criteria
- **Given** the same token sequence **when** `FakeRunner.prefill` runs twice
  **then** the readout logits are identical.
- **Given** a prefix cache **when** `prefill` is called with it **then** only
  the suffix tokens are counted in `usage.input_tokens` and `cached_tokens`
  equals the prefix length.
- **Given** a readout position list **when** `prefill` returns **then** logits
  exist only for those positions (no full-sequence vocab projection).

## Implementation notes
- The seam is three methods: `tokenize(text)`, `cache_prefix(tokens)`, and
  `prefill(tokens, prefix_cache=None, readout_positions=None)`. `prefill`
  takes the *full* sequence; with a cache it must start with the cached tokens
  and only the suffix is computed. Readout positions are absolute and must lie
  in the prefilled suffix (a cache holds KV, not logits). `check_prefill_args`
  is shared so every implementation enforces the same contract.
- `Readout` carries logits only (not hidden states plus an lm_head): the
  slice-before-projection happens inside the runner, which keeps the seam
  smaller for the Swift port. A logit row is anything indexable by token id
  (list, numpy row, `mx.array` row); readout only gathers the option ids.
- `FakeRunner` logits are a blake2b function of `tokens[:p+1]` and the token
  id, plus an optional static `logit_bias`, so cached and uncached prefills
  agree exactly. `ToyTokenizer` splits words longer than four letters into
  four-letter pieces so multi-token labels can be tested without a real
  tokenizer. `projected_rows` counts vocab projections.
