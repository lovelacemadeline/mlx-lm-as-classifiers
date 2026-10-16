---
type: story
status: completed
title: Decisions-per-second benchmark harness
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/bench.py::run_bench"
  - symbol: "src/mlx_classifiers/bench.py::environment"
  - symbol: "src/mlx_classifiers/cli.py::main"
  - test: "tests/test_bench.py::test_run_bench_sweeps_lengths_and_cache"
  - test: "tests/test_bench.py::test_filler_state_hits_the_token_budget"
---

# Story 2.4 — Decisions-per-second benchmark harness

A benchmark that reports decisions per second against state length, with and
without the prefix cache, INT8 on and off, and the prefill tok/s it implies.

## Acceptance Criteria
- **Given** a state-length sweep **when** the harness runs **then** it writes a
  table of decisions/s and prefill tok/s with the machine and power mode
  recorded.

## Implementation notes
- `mlx-classifiers bench [--model --lengths --reps --warmup --out]`
  (`[project.scripts]`, `cli.py`). It sweeps state length (repeated filler
  text trimmed by binary search to at most N state tokens) times prefix cached
  or not, for one 3-way `choice` question. It writes
  `bench_runs/bench-<timestamp>.md` and `.json` (gitignored) with the model,
  git revision, machine, mlx versions, and the power state read from
  `pmset -g` (`lowpowermode`) and the first line of `pmset -g batt`. It never
  changes the power state.
- A timed decision is `Runner.prefill` (from a prefix cache built once outside
  the loop, as a server would hold it) plus the masked-softmax answer.
  Compilation and binding are per cell, untimed. Implied prefill tok/s is
  tokens actually prefilled divided by p50 latency. The INT8 column is `off`
  until story 2.3 lands.
- Tested on the FakeRunner; the real run below used `MLXRunner` (story 2.2).

### Run 2026-10-16, Apple M5, low power mode on, AC power, git 98f1e3f, 20 reps

| state tok | prompt tok | prefix cached | prefilled tok | p50 ms | p95 ms | decisions/s | prefill tok/s |
|---|---|---|---|---|---|---|---|
| 128 | 248 | no | 248 | 105.7 | 117.0 | 9.5 | 2347 |
| 128 | 248 | yes | 182 | 89.6 | 92.7 | 11.2 | 2031 |
| 256 | 376 | no | 376 | 139.0 | 143.6 | 7.2 | 2704 |
| 256 | 376 | yes | 310 | 122.8 | 128.8 | 8.1 | 2524 |
| 512 | 632 | no | 632 | 208.5 | 213.1 | 4.8 | 3032 |
| 512 | 632 | yes | 566 | 203.7 | 240.4 | 4.9 | 2779 |
| 1024 | 1144 | no | 1144 | 355.3 | 363.3 | 2.8 | 3220 |
| 1024 | 1144 | yes | 1078 | 338.9 | 364.3 | 3.0 | 3180 |
| 2048 | 2168 | no | 2168 | 736.2 | 745.0 | 1.4 | 2945 |
| 2048 | 2168 | yes | 2102 | 726.1 | 796.6 | 1.4 | 2895 |

The benchmark prefix is only 66 tokens, so caching it saves little; the
per-decision floor (the L=1 pass through all 35 layers, story 2.5) dominates
short states. Implied tok/s matches 2.5's split-path prototype. Re-run with
low power mode off before quoting.
