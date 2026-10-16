---
type: story
status: completed
title: Fixture checker usable from another repo
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/fixtures.py::check"
  - symbol: "src/mlx_classifiers/fixtures.py::compare"
  - symbol: "src/mlx_classifiers/fixtures.py::CheckReport"
  - symbol: "src/mlx_classifiers/fixtures.py::CaseReport"
  - symbol: "src/mlx_classifiers/cli.py::main"
  - test: "tests/test_fixtures.py::test_one_wrong_token_id_names_position_and_both_ids"
  - test: "tests/test_fixtures.py::test_wrong_option_token_id_is_named"
  - test: "tests/test_fixtures.py::test_probability_tolerance"
  - test: "tests/test_fixtures.py::test_logits_reported_and_gated_only_on_request"
  - test: "tests/test_fixtures.py::test_missing_case_wrong_answer_and_error_mismatch_fail"
  - test: "tests/test_fixtures.py::test_port_that_answers_the_error_case_fails"
  - test: "tests/test_fixtures.py::test_cli_check_exit_codes"
  - test: "tests/test_fixtures.py::test_in_process_callable_results"
  - test: "tests/test_fixtures.py::test_committed_fixtures_reproduce_on_mlx"
---

# Story 4.2 — Fixture checker usable from another repo

A checker that takes a fixture directory and a callable (or a results JSON
produced by a port) and reports per-case token-id equality and probability
distance, so the Swift repo can run it in CI.

## Acceptance Criteria
- **Given** a results file with one wrong token id **when** checked **then** the
  case fails naming the position and both ids
  (`tokens[40]: expected token id 6937, got 6938`) and the exit code is 1.
- **Given** probabilities off by less than `--prob-tol` (default 1e-3) **then**
  the case passes and reports the max distance; beyond it, it fails naming the
  option.
- **Given** `--runner mlx` **then** the fixtures are regenerated in process on
  `MLXRunner` and compared (the determinism check); on this machine all 8 pass
  with distance 0.
- **Given** another repo **then** `mlx_classifiers.fixtures.check(dir, results)`
  accepts a results path, its parsed object, or a callable `fixture -> result`
  and returns a `CheckReport`.

## Implementation notes
- `mlx-classifiers fixtures check <dir> (--results R | --runner mlx)
  [--prob-tol] [--logit-tol] [--model]`. Results shape and comparison rules:
  [fixture-format](../reference/fixture-format.md).
- Exact: tokens, `prefix_len`, readout position, options, labels, option token
  ids, answer, `error.type`. Probabilities: max absolute distance. Raw logits:
  distance always reported, gated only with `--logit-tol`, because logits are
  path-dependent ([story 2.5](2-5-consumer-layers-on-readout-row.md)) and a
  port's kernels will not match bf16 bit for bit.
- Every `expected` key must be present in the result; an unexpected `error`
  fails; error messages are not compared.
