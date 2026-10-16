---
type: story
status: completed
title: Order permutation for position-bias reduction
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/decide.py::permutation_orders"
  - symbol: "src/mlx_classifiers/decide.py::decide"
  - test: "tests/test_permutation.py::test_two_permutations_average_two_bound_distributions"
  - test: "tests/test_permutation.py::test_position_bias_is_averaged_out"
  - test: "tests/test_permutation.py::test_permutation_orders_deterministic_and_valid"
---

# Story 1.7 — Order permutation for position-bias reduction

`options.permutations: N` compiles each question N times with option order
permuted (deterministic seed), averages probabilities, and reports the spread.

## Acceptance Criteria
- **Given** `permutations: 2` **when** a choice runs **then** two prompts are
  compiled, labels stay bound to options across orderings, and the answer is
  the mean of the two distributions.

## Implementation notes
- **Orderings.** `permutation_orders(n, N, seed)`: order 0 is the declared
  order, orders 1..n-1 are cyclic rotations (with `N <= n` every option sits
  at a different position each time), and beyond that `random.Random` shuffles
  seeded by `(question name, k)`. Fully deterministic.
- **Binding.** Each ordering is compiled separately; letters follow display
  position and `CompiledPrompt.label_for` maps them back to options, so the
  mean is taken in option order. Raw and calibrated distributions are averaged
  separately; `confidence` is the mean calibrated probability of the returned
  choice. A label-"A"-loving fake shows full rotation removes position bias
  exactly (uniform mean).
- **Scope.** Only `choice` questions are permuted. `noul` has a fixed Yes/No
  pair and `score` is an ordered scale whose order is part of the question,
  so both compile once whatever `permutations` says. Works in both readout
  modes; first-token orderings share the request's prefix cache.
- **Spread on the wire.** Reported as `debug.permutation_spread` (per option,
  max minus min calibrated probability), an additive x-local field added to
  `schema/systemone-response.schema.json` and the pydantic `Debug` model
  together (wire-schema "Versioning": additive x-local fields do not bump the
  version). `debug.option_token_ids` lists one label id per ordering in
  first-token mode.
