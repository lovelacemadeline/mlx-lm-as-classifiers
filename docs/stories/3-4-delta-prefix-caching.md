---
type: story
status: completed
title: Delta prefix caching on the longest common token prefix
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/prompt_cache.py::PromptCacheStore.lookup"
  - symbol: "src/mlx_classifiers/prompt_cache.py::PromptCacheStore.insert"
  - symbol: "src/mlx_classifiers/prompt_cache.py::without_delta"
  - symbol: "src/mlx_classifiers/mlx_runner.py::rewind_cache"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.prefill"
  - symbol: "src/mlx_classifiers/runner.py::FakeRunner.prefill"
  - symbol: "src/mlx_classifiers/schema.py::RequestOptions"
  - symbol: "src/mlx_classifiers/server.py::create_app"
  - symbol: "src/mlx_classifiers/bench.py::delta_states"
  - route: "POST /v1/systemone"
  - test: "tests/test_mlx_runner.py::test_delta_prefill_computes_only_the_delta_and_matches_uncached"
  - test: "tests/test_mlx_runner.py::test_rewind_after_the_ring_slid_is_exact_or_refused"
  - test: "tests/test_mlx_runner.py::test_hit_within_the_window_and_declared_prefix_fallback"
  - test: "tests/test_mlx_runner.py::test_ring_buffer_rewind_by_one_continues_exactly"
  - test: "tests/test_mlx_runner.py::test_chunked_prefill_keeps_the_last_chunk_full_for_rewinds"
  - test: "tests/test_prompt_cache.py::test_lru_evicts_by_entries_and_by_bytes"
  - test: "tests/test_prompt_cache.py::test_lookup_falls_back_when_a_rewind_is_not_representable"
  - test: "tests/test_prompt_cache.py::test_fake_runner_prefills_only_the_delta"
  - test: "tests/test_prompt_cache.py::test_decide_reuses_the_shared_head_across_questions_and_permutations"
  - test: "tests/test_prompt_cache.py::test_options_delta_cache_false_runs_without_the_store"
  - test: "tests/test_prompt_cache.py::test_transcript_steps_report_growing_cached_tokens"
  - test: "tests/test_server.py::test_server_reports_the_delta_hit_length"
  - test: "tests/test_server.py::test_server_flag_turns_the_delta_cache_off"
  - test: "tests/test_delta_cache_mlx.py::test_1024_state_with_a_50_token_tail_change_matches_uncached"
  - test: "tests/test_delta_cache_mlx.py::test_sliding_layers_rewound_past_the_window_match_uncached"
  - test: "tests/test_delta_cache_mlx.py::test_lru_bounds_entries_and_bytes_on_real_kv"
  - test: "tests/test_delta_cache_mlx.py::test_transcript_append_only_20_steps_grows_cached_and_stays_flat"
---

# Story 3.4 — Delta prefix caching

Before this story the server cached KV only for the declared `prefix` field.
For a sequential controller, most of the state repeats from tick to tick
(rules, map, inventory) and only a short tail changes; a transcript-style
memory grows by appending. Now the runner caches on the **longest common token
prefix** between the new request's full token sequence and any cached
sequence. It trims a copy of that cache to the shared length and prefills only
the delta. The hit length is reported in `usage.cached_tokens` and in the
`x-prefix-cache-tokens` header. The gridworld's transcript memory mode needs
this too.

How it works: [serving § Delta prefix caching](../reference/serving.md#delta-prefix-caching)
and the `mlx_runner` module docstring. In short:

- `PromptCacheStore` (`prompt_cache.py`) is a bounded LRU (entries and total
  KV bytes; default 8 and 2 GiB) keyed by token tuples. `lookup` tries entries
  in order of common-prefix length and takes the first one the runner can
  rewind. mlx-lm 0.31.3's `PromptTrie`/`LRUPromptCache` are not reused:
  `fetch_nearest_cache` considers only one longer and one shorter candidate
  and calls `RotatingKVCache.trim`, which is wrong once the ring has slid.
  The module docstring has the details.
- `MLXRunner.prefill` stores the state after the cache-only rows (`tokens[:p0]`),
  before the readout pass, so the stored sliding layers are in temporal order,
  never a ring.
- `rewind_cache`: `KVCache` trims to any length. A `RotatingKVCache` that never
  dropped a row is sliced. One that slid holds rows `[start, offset)` and can
  serve a hit `h` only if `h - start >= W - 1`. The rewound cache is those
  `W - 1` rows behind one placeholder row (`_idx = W`), which mlx-lm's concat
  path drops and its in-place path overwrites. Otherwise it returns None and
  the store tries the next entry, then the declared prefix.
- Cache-only prefill now runs its remainder chunk first, so the last chunk is
  a full `prefill_step`. A sliding layer keeps `W - 1` rows plus the last chunk,
  and that is the rewind budget of an entry built from a multi-chunk prefill.
  The first bench run found this: the 2048-token cell (prompt 2167, chunks
  2048 + 52) never hit.

## Acceptance Criteria
- **Given** two consecutive requests sharing the first N tokens **when** the
  second runs **then** `cached_tokens` is N and only the remainder is prefilled
  (asserted via the runner's recorded call shapes). Met:
  `test_delta_prefill_computes_only_the_delta_and_matches_uncached` (backbone
  calls `[(1, delta - 1), (1, 1)]`), `test_fake_runner_prefills_only_the_delta`.
- **Given** a 1024-token state with a 50-token tail change **when** benchmarked
  **then** decisions/s approaches the 128-token figure. **Partly met**: 5.2/s
  against 2.3/s uncached and 10.5/s for the 128-token delta cell (table
  below). The per-decision cost is about the delta plus an overhead that still
  grows with the cached length (not profiled cleanly: the GPU was busy
  afterwards).
- **Given** the RotatingKVCache sliding layers **when** a cache is trimmed to a
  prefix length beyond the window **then** results match an uncached prefill
  (argmax and probabilities within the fixture tolerance). Met on E2B at
  `DEFAULT_PROB_TOL` (1e-3), with hits past the window from a retained entry,
  from a slid entry within its delta, and a refused rewind that falls back.
  The toy model with mlx-lm's real cache classes matches to 1e-4.
- Additional: LRU eviction by entries and bytes (fake and real KV); a
  20-step append-only transcript (~100 tokens per step) reports cached tokens
  66, 170, 269 ... 1961 with flat per-step time (157 ms for the cold first
  step, then 111 to 127 ms); `options.delta_cache: false` and
  `serve --no-delta-cache` turn it off; the declared `prefix` keeps working.

## Bench (2026-10-16, M5 MBA, **low power mode on**, AC power)

`mlx-classifiers bench --lengths 128 256 512 1024 2048 --delta-tail 50`, E2B
(`gemma-4-E2B-it-qat-4bit`), git `3d235c2-dirty`, mlx 0.32.0, mlx-lm 0.31.3,
20 reps. The GPU was free for this run (no gridworld process). The delta
cell's state differs from the previous decision's in its last ~50 tokens.
Its prefill also covers the fixed question and answer turn after the state
(~48 tokens), so 98 tokens are prefilled per decision.

| state tok | prefix cached | delta tail | prefilled tok | cached tok | p50 ms | decisions/s |
|---|---|---|---|---|---|---|
| 128 | no | - | 248 | 0 | 110.7 | 9.0 |
| 128 | yes | - | 182 | 66 | 90.8 | 11.0 |
| 127 | yes | 50 | 98 | 150 | 95.6 | 10.5 |
| 256 | no | - | 376 | 0 | 139.3 | 7.2 |
| 256 | yes | - | 310 | 66 | 122.2 | 8.2 |
| 255 | yes | 50 | 98 | 278 | 101.0 | 9.9 |
| 512 | no | - | 632 | 0 | 216.2 | 4.6 |
| 512 | yes | - | 566 | 66 | 227.7 | 4.4 |
| 511 | yes | 50 | 98 | 534 | 124.7 | 8.0 |
| 1024 | no | - | 1144 | 0 | 433.4 | 2.3 |
| 1024 | yes | - | 1078 | 66 | 531.3 | 1.9 |
| 1023 | yes | 50 | 98 | 1046 | 190.8 | 5.2 |
| 2048 | no | - | 2168 | 0 | 1220.1 | 0.8 |
| 2048 | yes | - | 2102 | 66 | 1185.4 | 0.8 |
| 2047 | yes | 50 | **2102** | **66** | 1380.7 | 0.7 |

The last row is the multi-chunk miss described above. After the
remainder-first fix, `--lengths 1024 2048 --delta-tail 50` was re-run **with
the GPU shared** (the gridworld 26B game had started again), so compare its
rows with each other only:

| state tok | prefix cached | delta tail | prefilled tok | cached tok | p50 ms | decisions/s |
|---|---|---|---|---|---|---|
| 1024 | no | - | 1144 | 0 | 573.4 | 1.7 |
| 1023 | yes | 50 | 98 | 1046 | 175.7 | 5.7 |
| 2048 | no | - | 2168 | 0 | 1694.8 | 0.6 |
| 2047 | yes | 50 | 98 | 2070 | 185.4 | 5.4 |

**26B (`gemma-4-26B-A4B-it-qat-4bit`): not run, deferred to the
coordinator.** The GPU check passed before the E2B bench, but the gridworld
viewer (`gridworld play --backend local`, the 26B, about 15 GB wired) started
again right after it. The contended re-run above, and a final E2B
`pytest -m mlx` (20 passed; transcript steps 141 to 181 ms after a 253 ms cold
step), ran alongside it and pushed the machine into swap. Real-model work then
stopped. The 26B bench and any further E2B re-runs wait until
`pgrep -f "gridworld play"` returns nothing. Run it with
`mlx-classifiers bench --model mlx-community/gemma-4-26B-A4B-it-qat-4bit --lengths 128 512 1024 --delta-tail 50`.

## Notes and limits
- The delta path's numerics differ from the uncached path by bf16 chunking
  noise. On E2B first-token decisions stayed within 1e-3, but a sequence-mode
  golden fixture moved by 2.1e-3 probability (0.31 logit) when an earlier case
  gave it a hit. Fixture runs (`run_case`) therefore use `without_delta`,
  because a fixture pins its cache path.
- Sliding layers limit rewind depth. A tick that changes its state *earlier*
  than the previous tick did cannot use the previous tick's entry. It falls
  back to an older fully retained entry if one is still stored, else to the
  declared prefix (exact either way; `PromptCacheStore.unrepresentable`
  counts these). A gridworld whose agent moves around a map will hit this;
  [2.7](2-7-bounded-context-all-sliding.md) and snapshots are the longer-term
  answer.
- `x-prefix-cache` keeps its `hit | miss | none` meaning (the declared
  prefix): the gridworld's HTTP backend test compares it to `hit`. The delta
  hit length is the new `x-prefix-cache-tokens` header, not a change to the
  existing one.
- Each prefill now deep-copies its cache once more (the stored snapshot):
  0.3 ms measured. E2B KV is about 11 MB per 1k-token entry.
