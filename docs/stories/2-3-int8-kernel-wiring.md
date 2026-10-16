---
type: story
status: completed
title: INT8 kernel wiring into the runner
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/int8/wiring.py::Int8LinearW8A8"
  - symbol: "src/mlx_classifiers/int8/wiring.py::patch_mlps"
  - symbol: "src/mlx_classifiers/int8/wiring.py::patch_projs"
  - symbol: "src/mlx_classifiers/int8/wiring.py::patch_model"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner"
  - symbol: "src/mlx_classifiers/bench.py::run_bench"
  - test: "tests/test_int8.py::test_fixtures_agree_int8_on_vs_off"
  - test: "tests/test_int8.py::test_load_patches_every_mlp_and_projection"
  - test: "tests/test_int8.py::test_wrapper_keeps_bf16_and_falls_back_under_64_rows"
  - test: "tests/test_int8.py::test_bench_int8_column_follows_the_runner_flag"
---

# Story 2.3 — INT8 kernel wiring into the runner

Vendor `w8a8_wiring.py` from `aug/poorly-ported-gemma4-int8-optimisations`
(MLP + projection W8A8, self-contained) into `src/mlx_classifiers/int8/` and
wire it behind a flag, with a parity test against the stock path. The
attention router (`agent_e4b_wire4.py`) is a follow-up (story 2.6) because of
its length-multiple-of-64 and batch-1 constraints. See the
[recon](../research/e2b-int8-and-vlm-recon-2026-10.md) §A, §C.

## Acceptance Criteria
- **Given** INT8 on and off **when** the same fixtures run **then** argmax
  agrees on every case and probabilities agree within a stated tolerance.
  Tolerance: raw probabilities within 5e-3, raw logits within 1.0
  (`tests/test_int8.py::PROB_TOL`, `LOGIT_TOL`).
- **Given** `MLXRunner.load(repo, int8=True)` **when** loaded on Gemma 4 E2B
  **then** all 105 MLP linears and all 100 attention projections (15 producer
  layers x 4, 20 KV-shared layers x 2: q and o only) are W8A8 wrappers; the
  default is `int8=False`.
- **Given** an input under 64 rows **when** a wrapper runs **then** the output is
  bit-identical to the stock linear; above it the output stays bf16.
  *Superseded by [story 2.9](2-9-memory-budget.md):* the stock linear is no
  longer kept. Calls of up to 8 rows run a GEMV on the int8 weights, and 9 to
  63 rows are padded into the int8 GEMM, within 5% of stock (same test).
- **Given** `mlx-classifiers bench --int8` **when** run **then** the int8 column
  reads `on`.

## Implementation notes

- `src/mlx_classifiers/int8/wiring.py` is the source file verbatim (sha256
  `7f7be111...`, mtime 2026-08-09) under a provenance header. The only change is
  in the patch entry points: the layer walk goes through `_decoder_layers`
  (gemma4 wrapper or bare text model), and `patch_model` applies both patches.
  The kernel compiled and ran first time on E2B (M5, macOS 26.4, Metal 4), no
  numerical changes needed.
- The "qat-4bit" checkpoint stores these linears as **8-bit, group 64** affine;
  `Int8LinearW8A8` dequantises that and requantises per output channel.
- `MLXRunner.load(..., int8=True)` applies `patch_model` after `mlx_lm.load`
  and sets `runner.int8`. `bench --int8` and `fixtures check --runner mlx
  --int8` pass the flag through; the bench's int8 column reads `runner.int8`.
- **Memory:** resident (MLX active) 3.38 GB stock, 5.22 GB with INT8. The
  wrapper keeps the stock linear for the under-64-row fallback, so the int8
  weights are added on top. Patching takes about 8.5 s at load (numpy
  requantisation). Dropping the stock copy would need a stock-equivalent
  fallback for L=1; left for later. (Done in [story 2.9](2-9-memory-budget.md):
  3.41 GB with INT8.)
- Where INT8 engages: the cache-only prefill and the sequence-mode readout
  span when they are 64 rows or more. The L=1 first-token readout is stock.
  `noul-sequence` (55 tokens) never reaches the int8 lane.

### Parity (2026-10-16, low power mode on, AC)

Stock reproduces the committed fixtures exactly (dp 0). INT8 on, against them:
argmax agrees on all 7 scored cases (the error case still errors). Max |dp|
1.44e-3 (`score-sequence-object-state`), max |dlogit| 0.5 (bf16 steps).
Bench prompts at 632 to 4216 tokens: argmax agrees, max |dp| 1.6e-3, max
|dlogit| 0.75. The 5e-3 tolerance is about 3x the worst measured case and under
1% of the smallest top-1 margin in the fixtures (0.934), so it cannot hide a
flip. 1.0 for logits matches the cross-path bf16 noise of story 2.5. These
deltas fit int8-mlx story 9.4's 0.981 top-1 fidelity, since these fixtures have
wide margins.

### Bench, Apple M5, low power mode on (`lowpowermode 1`), AC power, git c432ade-dirty, 20 reps

INT8 off:

| state tok | prompt tok | prefix cached | int8 | prefilled tok | p50 ms | p95 ms | decisions/s | prefill tok/s |
|---|---|---|---|---|---|---|---|---|
| 128 | 248 | no | off | 248 | 105.9 | 113.1 | 9.4 | 2342 |
| 128 | 248 | yes | off | 182 | 89.7 | 95.4 | 11.2 | 2030 |
| 256 | 376 | no | off | 376 | 140.6 | 148.6 | 7.1 | 2674 |
| 256 | 376 | yes | off | 310 | 123.5 | 135.8 | 8.1 | 2511 |
| 512 | 632 | no | off | 632 | 209.8 | 213.9 | 4.8 | 3013 |
| 512 | 632 | yes | off | 566 | 192.8 | 204.6 | 5.2 | 2936 |
| 1024 | 1144 | no | off | 1144 | 356.9 | 361.4 | 2.8 | 3205 |
| 1024 | 1144 | yes | off | 1078 | 340.8 | 367.5 | 2.9 | 3163 |
| 2048 | 2168 | no | off | 2168 | 741.6 | 818.8 | 1.3 | 2924 |
| 2048 | 2168 | yes | off | 2102 | 731.0 | 735.4 | 1.4 | 2875 |

INT8 on (128 to 512 rows from the full sweep; the full sweep's 1024 and 2048
rows were noisy, p95 up to 1.6 s, so the rows below are a `--lengths 1024 2048`
rerun, with a stock rerun alongside within 0.5% of the table above):

| state tok | prompt tok | prefix cached | int8 | prefilled tok | p50 ms | p95 ms | decisions/s | prefill tok/s |
|---|---|---|---|---|---|---|---|---|
| 128 | 248 | no | on | 248 | 91.8 | 98.1 | 10.9 | 2701 |
| 128 | 248 | yes | on | 182 | 80.9 | 83.7 | 12.4 | 2250 |
| 256 | 376 | no | on | 376 | 118.7 | 121.0 | 8.4 | 3167 |
| 256 | 376 | yes | on | 310 | 105.1 | 107.5 | 9.5 | 2949 |
| 512 | 632 | no | on | 632 | 178.7 | 186.8 | 5.6 | 3537 |
| 512 | 632 | yes | on | 566 | 165.9 | 170.0 | 6.0 | 3411 |
| 1024 | 1144 | no | on | 1144 | 290.6 | 294.7 | 3.4 | 3937 |
| 1024 | 1144 | yes | on | 1078 | 275.4 | 279.1 | 3.6 | 3914 |
| 2048 | 2168 | no | on | 2168 | 574.1 | 602.7 | 1.7 | 3776 |
| 2048 | 2168 | yes | on | 2102 | 598.1 | 693.0 | 1.7 | 3514 |

INT8 gives 1.11x to 1.16x decisions/s at 128 to 512 state tokens and about
1.2x to 1.3x at 1k to 2k tokens. The recon's 1.3 to 1.4x for MLP plus
projections was measured on long prefills with low power mode off. The L=1
readout floor (story 2.5) is still stock, which caps the gain on short prompts.
