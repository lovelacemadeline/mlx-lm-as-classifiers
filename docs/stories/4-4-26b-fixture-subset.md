---
type: story
status: completed
title: 26B fixture subset — five seed cases pinned to Gemma 4 26B-A4B in fixtures/26b
owner: maddie
updated: 2026-10-17
anchor:
  - symbol: "src/mlx_classifiers/fixtures.py::select_cases"
  - symbol: "src/mlx_classifiers/fixtures.py::add_arguments"
  - symbol: "src/mlx_classifiers/fixtures.py::generate"
  - test: "tests/test_fixtures.py::test_select_cases_keeps_the_given_order_and_rejects_unknown_ids"
  - test: "tests/test_fixtures.py::test_generate_cases_flag_writes_only_the_subset"
  - test: "tests/test_fixtures.py::test_committed_26b_fixtures_are_seed_cases_pinned_to_the_26b"
  - test: "tests/test_fixtures.py::test_committed_26b_fixtures_pass_on_mlx"
---

# Story 4.4 — 26B fixture subset — five seed cases pinned to Gemma 4 26B-A4B in fixtures/26b

The Swift port (mlx-swift-as-classifiers, "26B on Swift") needs a contract for
the model that matters, `mlx-community/gemma-4-26B-A4B-it-qat-4bit`, without
doubling the E2B set. `mlx-classifiers fixtures generate` gains
`--cases id,id,...` (default: every seed case; an unknown id exits 2), and the
26B set lives in its own directory, `fixtures/26b/`, so `fixtures check` keeps
taking the model from the first fixture and the flat E2B set (`fixtures/*.json`,
16 cases) is untouched. Every fixture already carries `model.repo` per case, so
a port can skip a model it lacks.

Generated on the stock 4-bit path (no `--int8`), `moe_kernel` at its default
(on), split prefill step 2048, on 2026-10-17 with low-power mode on
(`pmset lowpowermode 1`); generate (load + five cases) took 16.7 s wall.

| case | gate | answer | top-2 logit gap |
|---|---|---|---|
| noul-first-token | full | yes | 17.9 |
| noul-sequence | full | no | 15.5 |
| choice-sequence-multi-token-options | full | refund (p 0.994) | 5.1 |
| choice-first-token-image-gridworld-vt140 | answer-and-ids | south (p 0.398 vs east 0.265) | 0.41 |
| choice-first-token-image-gridworld-you-vt140 | full | blue (p 1.0) | 11.7 |

The gridworld action case is near-tied on the 26B as on the E2B, so it keeps
its seed gate (`answer-and-ids`); the clear-winner image case is gated in full.
`fixtures check fixtures/26b --runner mlx` passes 5/5 with max|dp| 0.

## Acceptance Criteria
- **Given** `--cases noul-sequence,noul-first-token` **when** cases are
  selected **then** exactly those come back in that order, and an unknown id
  raises (the CLI exits 2).
- **Given** `fixtures generate out --cases noul-sequence` **when** it runs
  **then** `out/` holds only `noul-sequence.json`.
- **Given** the committed `fixtures/26b/` **when** it is loaded **then** it is
  the five cases above, each with `model.repo` the 26B and its seed gate, and
  `fixtures/` alone still loads only E2B fixtures.
- **Given** the 26B snapshot in the HF cache **when** `fixtures/26b/` is
  checked with MLXRunner **then** every case passes (skipped without the
  snapshot).

## Related
- [Fixture format](../reference/fixture-format.md)
- [Story 4.3 — per-fixture gate](4-3-per-fixture-gate-and-clear-winner-image-fixtures.md)
