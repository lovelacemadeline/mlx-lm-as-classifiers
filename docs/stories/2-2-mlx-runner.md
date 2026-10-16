---
type: story
status: completed
title: MLX runner — load, cached prefix, prefill, readout-position slice
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner"
  - symbol: "src/mlx_classifiers/runner.py::LogitRow"
  - test: "tests/test_mlx_runner.py::test_split_runs_cache_only_rows_then_one_readout_pass"
  - test: "tests/test_mlx_runner.py::test_multi_readout_projects_only_gathered_rows"
  - test: "tests/test_mlx_runner.py::test_prefix_cache_is_copied_not_mutated"
  - test: "tests/test_mlx_integration.py::test_three_question_types_answer_the_obvious"
  - test: "tests/test_mlx_integration.py::test_raw_probabilities_identical_across_runs"
  - test: "tests/test_mlx_integration.py::test_prefix_longer_than_sliding_window"
  - test: "tests/test_mlx_integration.py::test_2k_prompt_projects_only_the_readout_row"
---

# Story 2.2 — MLX runner — load, cached prefix, prefill, readout-position slice

Implement `Runner` (story 1.2) on mlx-lm for Gemma 4 E2B: load, build and
reuse a prefix KV cache, prefill the suffix, and slice the final hidden state
to the readout positions **before** the lm_head. Depends on stock PyPI mlx-lm 0.31.3 and mlx 0.32.0 per the [recon](../research/e2b-int8-and-vlm-recon-2026-10.md); no fork. Story 2.5 owns the KV-consumer trick; this story may run the full forward first and must leave the seam for 2.5 (Gemma vocab is large; the full
projection is the dominant cost at long prompts).

## Acceptance Criteria
- **Given** a request with a cached prefix **when** run **then** only suffix
  tokens are prefilled and `cached_tokens` reports the prefix length.
- **Given** a 2k-token prompt **when** profiled **then** no vocab-sized
  projection runs over positions other than the readout positions.
- **Given** the same request **when** run twice **then** raw probabilities are
  identical.

## Implementation notes
- **Split prefill, generalised** (the port seam; full description in the
  `mlx_runner` module docstring). Rows before `min(readout_positions)` run
  cache-only (backbone called, only `c.state` evaluated, so MLX prunes the 20
  KV-consumer layers and the lm_head); rows `min..max` run one full forward;
  the hidden state is gathered at the readout positions *before* the tied
  lm_head and the final logit soft-cap; rows after the last readout never
  run. This covers single readout, multi-readout and sequence mode with no
  `model(all)[:, -1]`. Cache-only rows are chunked at 2048 like mlx-lm's
  `generate_step`. It reuses the generate-style split that story 2.5's
  prototype measured; 2.5's remaining packed-questions scope is untouched.
- **Prefix cache** = the mlx-lm cache list after a cache-only prefill;
  `prefill` deep-copies it. Tested with a prefix over 600 tokens (past the
  512-token `RotatingKVCache` window): cached, re-cached and uncached logits
  agree within bf16 noise on the top 20 ids with the same argmax, and the
  shared cache offsets are unchanged.
- **Rows.** `LogitRow` became one concrete class returned by every runner
  (FakeRunner computes on demand; MLXRunner wraps the evaluated float32
  `mx.array` row and its `mx.logsumexp`), so sequence mode gets a vectorised
  normaliser. The `input_tokens` of a prefill counts rows actually run (up to
  the last readout position).
- `mlx`/`mlx_lm` are imported lazily; `MLXRunner.load(repo)` defaults to
  `mlx-community/gemma-4-E2B-it-qat-4bit`; `chat_template` is
  `HFChatTemplate` (thinking off).
- **Real-model check** (low power mode on): on a payments outage ticket
  `about_payments` P(yes) is 0.995, `department` is billing at 0.98 and
  `severity` is critical at 0.997, in both first-token and sequence modes;
  an unrelated state gives P(yes) below 0.2. Raw probabilities are identical
  across runs. Integration tests carry the `mlx` marker and skip when the
  weights are not cached.
- `decide`'s `timing_ms.readout` is total minus prefill, so it includes prompt
  compilation and the one-off `cache_prefix`.
