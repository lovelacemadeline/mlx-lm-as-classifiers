---
type: story
status: completed
title: Fixture format and generator
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/fixtures.py::generate"
  - symbol: "src/mlx_classifiers/fixtures.py::run_case"
  - symbol: "src/mlx_classifiers/fixtures.py::build_fixture"
  - symbol: "src/mlx_classifiers/fixtures.py::model_provenance"
  - symbol: "src/mlx_classifiers/fixtures.py::FixtureCase"
  - symbol: "src/mlx_classifiers/fixtures.py::load_fixtures"
  - test: "tests/test_fixtures.py::test_generate_twice_is_byte_identical"
  - test: "tests/test_fixtures.py::test_seed_set_covers_types_modes_cache_state_shapes_and_an_error"
  - test: "tests/test_fixtures.py::test_fixture_pins_provenance_path_and_readout"
  - test: "tests/test_fixtures.py::test_cached_case_reuses_the_system_turn"
  - test: "tests/test_fixtures.py::test_multi_token_label_case_records_error_and_label_ids"
  - test: "tests/test_fixtures.py::test_committed_fixtures_are_the_seed_set_on_gemma"
  - test: "tests/test_fixtures.py::test_generate_on_mlx_is_byte_identical"
---

# Story 4.1 — Fixture format and generator

A fixture is one JSON file: request, model id and revision, tokenizer hash,
compiled token ids, readout position, option token ids, raw logits at the
mask, raw probabilities, chosen answer, readout mode. A generator produces
them from the Python runner. The shape is documented in
[fixture-format](../reference/fixture-format.md).

## Acceptance Criteria
- **Given** the generator **when** run twice on the same machine **then** the
  fixture files are byte-identical (FakeRunner in the fast suite; MLXRunner on
  Gemma 4 E2B in the `mlx`-marked test, and across two CLI processes).
- **Given** `mlx-classifiers fixtures generate <out_dir>` **when** it runs
  **then** it writes one `<id>.json` per seed case through `MLXRunner` on
  `mlx-community/gemma-4-E2B-it-qat-4bit`, each pinning the HF snapshot
  revision, the sha256 of `tokenizer.json`, and the split-prefill path.
- **Given** the seed set **then** it covers noul, choice and score in both
  readout modes, a prefix-cached case, a JSON-object state and a list state,
  and a multi-token-label case whose expected result is a
  `LabelTokenizationError` with each label's in-context token ids.

## Implementation notes
- `src/mlx_classifiers/fixtures.py`: `SEED_CASES` (8 `FixtureCase`s, each a
  one-question wire request plus `prefix_cached` and `label_style`),
  `run_case` (compile, bind in context, prefill, gather, raw softmax; the
  result block a port mirrors), `build_fixture`, `generate`,
  `model_provenance` (snapshot hash and tokenizer sha256 from the local HF
  cache via `try_to_load_from_cache`), `load_fixtures`. CLI entry is
  `fixtures generate` in `cli.py`.
- Raw probabilities are T = 1; no calibration temperature and no permutations
  enter a fixture.
- The multi-token-label case renders option texts as labels
  (`label_style: "option-text"`, the sequence-mode prompt) and reads them out
  first-token. Since story 1.8 it does so through the wire option
  `options.label_style`, not a private compile. On Gemma `chargeback` is two tokens `[10690, 1942]`, so binding
  raises. Letters and Yes/No are always single tokens, so the normal compile
  cannot reach this error.
- Fixtures pin the split path, never the naive full forward
  ([story 2.5](2-5-consumer-layers-on-readout-row.md)). On the cached seed
  case, cached and uncached split prefill gave identical logits; the boundary
  is recorded anyway.
- Generated set: `fixtures/` (8 files, 32 KB), revision
  `42f62737af7a9fd8c1d55d79666c1a217be4e2e2`. Generation takes about 5 s
  including the model load. Answers are sane (payout ticket -> yes; double
  charge -> billing; crash transcript -> technical; plants -> no).
