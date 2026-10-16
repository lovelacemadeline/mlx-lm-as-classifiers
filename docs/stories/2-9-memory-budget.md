---
type: story
status: in-progress
title: Memory budget — stop keeping two copies of INT8-patched weights, then quantise the KV store
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/memory.py::run_audit"
  - symbol: "src/mlx_classifiers/memory.py::Audit"
  - symbol: "src/mlx_classifiers/memory.py::process_memory"
  - symbol: "src/mlx_classifiers/memory.py::tree_nbytes"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.load"
  - symbol: "src/mlx_classifiers/int8/wiring.py::Int8LinearW8A8"
  - symbol: "src/mlx_classifiers/int8/wiring.py::_gemv_int8"
  - symbol: "src/mlx_classifiers/int8/wiring.py::_eligible"
  - symbol: "src/mlx_classifiers/int8/wiring.py::patch_model"
  - symbol: "src/mlx_classifiers/int8/experts.py::Int8SwitchLinear"
  - symbol: "src/mlx_classifiers/int8/experts.py::patch_experts"
  - test: "tests/test_memory.py::test_audit_renders_each_stage_and_its_change_from_the_previous_one"
  - test: "tests/test_memory.py::test_process_memory_reads_resident_and_footprint_on_macos"
  - test: "tests/test_memory.py::test_tree_nbytes_counts_underscored_arrays_once"
  - test: "tests/test_memory.py::test_cli_has_a_memory_command"
  - test: "tests/test_memory.py::test_int8_load_holds_no_stock_copy_and_stays_within_10pct"
  - test: "tests/test_int8.py::test_small_rows_run_on_the_int8_weights"
  - test: "tests/test_int8.py::test_wrapper_keeps_bf16_and_falls_back_under_64_rows"
  - test: "tests/test_int8.py::test_fixtures_agree_int8_on_vs_off"
  - test: "tests/test_int8_experts.py::test_26b_int8_experts_agree_with_stock_on_fixtures_and_delta_decisions"
---

# Story 2.9 — Memory budget

The 26B with everything on sits well above its 15 GB of 4-bit weights, and
with a viewer and an agent sharing the machine the box swaps. Measured
pieces so far: the INT8 wrappers keep the stock linear beside the int8 copy
for the under-64-row fallback (E2B went 3.4 to 5.2 GB resident in story 2.3;
the 26B expert path added about 1.4 GB in 2.8); the vendored vision tower;
the delta cache store (2 GiB cap by default, story 3.4); a separate planner
cache; and, when `--plan-model` is used, a second model entirely. Maddie's
earlier gemma4-turboquant-mlx fork ran four 128k streams of the 26B on this
machine with PolarQuant 2-bit KV, so the KV side has a known answer; the
weight duplication does not.

In order:
1. **Audit**: a `mlx-classifiers memory` report that loads a model with each
   flag and prints resident and MLX active/peak memory per component. **Done.**
2. **Drop the duplicate weights**: make the INT8 fallback path run on the
   int8 weights (pad rows to the kernel's minimum, or dequantise on the fly
   for small rows) so the stock linears can be freed after patching. **Done.**
3. **Quantise the KV store**: vendor the PolarQuant 2-bit codec from
   gemma4-turboquant-mlx for stored (not live) cache entries, so the delta
   store holds many more sequences in the same budget; measure the effect
   on probabilities against the fixtures. **Not started** (follow-up below).
4. Size the delta store cap by model and expose it on the CLI and server.
   **Not started** (follow-up below).

## Acceptance Criteria
- **Given** the audit **when** run for E2B and the 26B with int8 and vision
  **then** each component's bytes are listed. **Met**: tables below, from
  `mlx-classifiers memory` (`held` is each component's array bytes, counted
  directly; `tests/test_memory.py`).
- **Given** step 2 **when** the 26B loads with `int8=True` **then** resident
  memory is within 10% of the stock load and the parity gate still passes.
  **Met**: MLX active 14.47 GB stock, 14.92 GB INT8 (+3.1%; was 17.54, +21%);
  footprint after one decision 17.34 stock, 17.94 INT8 (+3.5%; was 22.85).
  `test_int8_load_holds_no_stock_copy_and_stays_within_10pct` asserts the active
  bound on E2B and the 26B; the 26B and E2B parity gates pass at their existing
  tolerances.
- **Given** step 3 **when** the delta bench runs with a 20-step transcript
  **then** store bytes drop by at least 4x with argmax unchanged on fixtures.
  **Open** (step 3).

## Step 1 — the audit (`mlx-classifiers memory`)

`mlx-classifiers memory --model REPO [--int8] [--vision] [--delta-store-mb N]
[--store-entries K] [--store-tokens L]` (`src/mlx_classifiers/memory.py`) loads
the model the way the server does and samples after each stage. The stages are
base weights, then the INT8 patch, then the vision tower, then a delta store
warmed by K decisions (default 4) over states of L tokens (default 1024) that
differ in their last 50, then one more decision. `MLXRunner.load(on_stage=...)`
is the hook between the load stages. Columns:

- **footprint**: the macOS physical footprint from `task_info(TASK_VM_INFO)`,
  what Activity Monitor shows. It includes Metal buffers and MLX's cache.
- **rss**: resident size from the same call. It does **not** track Metal
  buffers on the 26B (it falls while active memory rises), so read footprint.
- **active / cache / peak**: `mx.get_active_memory()`,
  `mx.get_cache_memory()`, and `mx.get_peak_memory()`, with peak reset after
  every sample, so it is that stage's high-water mark.
- **max_rss**: the `getrusage` maximum so far.
- **held**: the bytes of the arrays the stage added (model tree including
  underscored attributes, vision tower, store entries).

Every value is in GB, and each `d` column is the change from the previous stage.

Run 2026-10-16, M5 MBA (32 GB), lowpowermode 0, AC power ("Now drawing from 'AC
Power'", 93% charging), mlx 0.32.0, mlx-lm 0.31.3, git `1b7dcd3` plus this
story's audit command. The GPU was free (no `gridworld play`). `vm_stat`
swapouts went 27919258 -> 27925522 over the eight before-runs. The +6264 came
during the 26B INT8 + vision runs, whose footprint reached 24 GB.

### Before step 2

E2B (`gemma-4-E2B-it-qat-4bit`), `--int8 --vision`:

| stage | footprint | d | active | d | cache | peak | held |
|---|---|---|---|---|---|---|---|
| base weights | 3.80 | +3.77 | 3.38 | +3.38 | 0.00 | 3.38 | 3.38 |
| int8 patch | 6.21 | +2.41 | 5.22 | +1.84 | 0.15 | 5.25 | 1.84 |
| vision tower | 6.54 | +0.34 | 5.56 | +0.34 | 0.15 | 5.56 | 0.34 |
| delta store (4 x 1144 tok) | 7.50 | +0.95 | 5.63 | +0.07 | 1.01 | 6.18 | 0.07 |
| one decision | 7.51 | +0.01 | 5.63 | +0.00 | 1.01 | 5.76 | - |

26B (`gemma-4-26B-A4B-it-qat-4bit`), `--int8 --vision`:

| stage | footprint | d | active | d | cache | peak | held |
|---|---|---|---|---|---|---|---|
| base weights | 14.89 | +14.86 | 14.47 | +14.47 | 0.00 | 14.47 | 14.47 |
| int8 patch | 20.28 | +5.39 | 17.54 | +3.08 | 1.79 | 17.74 | 3.08 |
| vision tower | 21.42 | +1.14 | 18.69 | +1.14 | 1.79 | 18.69 | 1.14 |
| delta store (4 x 1148 tok) | 23.76 | +2.33 | 19.72 | +1.03 | 3.06 | 19.96 | 1.03 |
| one decision | 23.99 | +0.23 | 19.72 | +0.00 | 3.29 | 20.06 | - |

The runs without `--int8` or `--vision` match these rows stage for stage (the
stages they share agree within 0.01 GB). The end states are in the
before/after table below. What the audit found:

- **INT8 doubled the dense weights.** On E2B the int8 copies are 1.84 GB
  on top of 1.81 GB of stock linears. On the 26B it is 1.64 GB of int8 dense
  weights on top of 1.19 GB of stock, plus 1.43 GB from story 2.8's fp32 fold
  operands (scales and `b + 8s` widened from bf16).
- **The 26B's freed stock experts stayed in MLX's cache.** `patch_experts`
  drops the 11.4 GB of stock expert arrays, but nothing returned their buffers
  to the OS. They sat in MLX's cache (+1.79 GB footprint) until a later
  allocation reused them.
- **Component sizes.** The vision tower is 0.34 GB on E2B and 1.14 GB on the
  26B. A delta store entry at about 1146 tokens is 18 MB on E2B (KV sharing:
  15 of 35 layers own KV) and 258 MB on the 26B. The default 2 GiB cap
  therefore holds about 8 entries on the 26B and is never the binding limit on
  E2B; the 8-entry cap is.
- **MLX's buffer cache.** After the store's prefills it holds 0.86 GB (E2B)
  to 1.5 GB (26B) of transients. That is reclaimable but counted in the
  footprint, so `mx.set_cache_limit` is the lever (step 4).

## Step 2 — the int8 small-row path, stock copies freed

`Int8LinearW8A8` no longer keeps `orig`. Its calls now go:

- **1 to 8 rows** (`GEMV_MAX_ROWS`; the L=1 readout): `_gemv_int8`, a GEMV
  over the int8 weights dequantised on the fly. Activations stay bf16, the
  sum is fp32, and the output is bf16. Each simdgroup owns 2 output rows, and
  each lane reads 16 contiguous int8 of a row per step. Against the CPU fp32
  product of the same int8 weights it is within one bf16 rounding.
- **9 to 63 rows**: zero-padded to 64 and run through the existing W8A8 GEMM
  (int8 activations).
- **64 rows and over**: unchanged, and bit-identical to before.

To make the GEMV fast, the int8 weight is now stored `(N, K)`, not
pre-transposed `(K, N)`. The GEMM reads it with `transpose_right`, the way
story 2.8's expert kernel reads its nibbles. Its output is bit-equal to the old
layout at 64, 200 and 1024 rows on seven E2B and 26B shapes. The other
changes:

- **Patching.** `patch_mlps` and `patch_projs` only wrap linears the kernels
  can serve (`_eligible`: affine quantised, no bias, K % 32, N % 64; every
  E2B and 26B linear qualifies). `patch_model` and `patch_experts` call
  `mx.clear_cache()`.
- **Requantisation.** It runs in mx on the GPU instead of numpy. The int8
  weights and scales are bit-equal on all 205 E2B linears, the patch time
  drops (2.1 s -> 1.0 s on E2B's linears), and there are no host fp32 copies.
- **Expert fold operands.** `Int8SwitchLinear` keeps the checkpoint's bf16
  scales and biases (transposed) and forms `b + 8s` in fp32 in the kernel.
  Since `8s` is exact, this rounds once, as the precompute did. `Int8SwitchGLU`
  outputs are bit-identical to before at 1, 7, 64 and 300 rows. This is
  story 2.8's +1.4 GB.

### Which small-row path: measured on the readout pass

`proto/readout_l1.py MODEL 512 100 int8 [pad]` times the L=1 backbone pass
over the bench prompt's readout row (632/636-token prompt, fresh cache copy
per rep, median of 100 after 5 warm). Same conditions as above.

| small-row path (INT8 on) | E2B ms | 26B ms |
|---|---|---|
| stock linear kept (before; HEAD `1b7dcd3`) | 16.96 | 29.85 |
| pad to 64 rows, int8 GEMM | 37.44 | 44.51 |
| GEMV over `(K, N)` int8, K split into fp32 partials | 26.35 | 40.51 |
| GEMV over `(N, K)` int8, 4 rows per simdgroup | 18.44 | 33.13 |
| **GEMV over `(N, K)` int8, 2 rows per simdgroup (chosen)** | **17.92** | **32.59** |

Dequantising on the fly wins, and padding to 64 loses badly at L=1: each
64x64 tile is one simdgroup streaming all of K, so a short call fills only a
few dozen threadgroups. The chosen row was re-measured against HEAD in three
interleaved pairs (HEAD 17.11-17.13 / 29.82-29.93 ms; this story
17.91-17.94 / 32.53-32.60 ms). That is **+4.7% on E2B and +9.2% on the 26B**,
inside the 10% gate. The 26B's remainder is bytes, not the kernel: its
attention projections are **4-bit** in the checkpoint, so their int8 copy
reads 0.45 GB more per pass than stock. 1 row per simdgroup, 32 bytes per lane
and 2 or 8 simdgroups per threadgroup measured the same as the chosen row or
worse (32.35-33.25 ms on the 26B).

Prefill: `bench --int8 --lengths 2048` (E2B, uncached, 10 reps, alternating)
measured HEAD 293.6 / 287.5 ms against 296.4 / 296.7 ms with this story, about
+1 to 3%. At 512 tokens it was 84.5 -> 86.2 ms. The readout row accounts for
about 0.8 ms of that. The rest is the transposed B read in the GEMM.

### After step 2

E2B, `--int8 --vision`:

| stage | footprint | d | active | d | cache | peak | held |
|---|---|---|---|---|---|---|---|
| base weights | 3.79 | +3.77 | 3.38 | +3.38 | 0.00 | 3.38 | 3.38 |
| int8 patch | 4.55 | +0.76 | 3.41 | +0.03 | 0.00 | 3.64 | 0.03 |
| vision tower | 4.23 | -0.32 | 3.75 | +0.34 | 0.00 | 3.75 | 0.34 |
| delta store (4 x 1144 tok) | 5.14 | +0.91 | 3.81 | +0.07 | 0.86 | 4.37 | 0.07 |
| one decision | 5.15 | +0.01 | 3.82 | +0.00 | 0.86 | 3.95 | - |

26B, `--int8 --vision`:

| stage | footprint | d | active | d | cache | peak | held |
|---|---|---|---|---|---|---|---|
| base weights | 14.89 | +14.86 | 14.47 | +14.47 | 0.00 | 14.47 | 14.47 |
| int8 patch | 15.61 | +0.72 | 14.92 | +0.45 | 0.00 | 15.26 | 0.45 |
| vision tower | 16.52 | +0.91 | 16.06 | +1.14 | 0.00 | 16.06 | 1.14 |
| delta store (4 x 1148 tok) | 18.85 | +2.33 | 17.09 | +1.03 | 1.27 | 17.33 | 1.03 |
| one decision | 19.08 | +0.23 | 17.10 | +0.00 | 1.49 | 17.44 | - |

The footprint lags the frees right after the patch (+0.7 GB with active
+0.03 GB). Those pages come back during the next stages: the E2B vision stage
reads -0.32 GB.

### Memory before and after, end state (after one decision)

| model | flags | footprint before | footprint after | active before | active after |
|---|---|---|---|---|---|
| E2B | stock | 4.79 | 4.79 | 3.45 | 3.45 |
| E2B | --vision | 5.12 | 5.12 | 3.79 | 3.79 |
| E2B | --int8 | 7.17 | 4.81 | 5.29 | 3.48 |
| E2B | --int8 --vision | 7.51 | 5.15 | 5.63 | 3.82 |
| 26B | stock | 17.34 | 17.34 | 15.50 | 15.50 |
| 26B | --vision | 18.48 | 18.48 | 16.64 | 16.64 |
| 26B | --int8 | 22.85 | 17.94 | 18.58 | 15.95 |
| 26B | --int8 --vision | 23.99 | 19.08 | 19.72 | 17.10 |

INT8 now costs +0.03 GB of weights on E2B (was +1.84) and +0.45 GB on the 26B
(was +3.08). The 26B with everything on is 4.9 GB smaller in footprint. Over
the after-runs, `vm_stat` swapouts stayed at 28146922.

### Parity (fixtures, INT8 on; HEAD `1b7dcd3` vs this story)

E2B against the committed (stock) fixtures. The gate is |dp| 5e-3 and
|dlogit| 1.0 for cases whose top-1 margin is at least 0.5. The three low-margin
gridworld frames are gated on token ids and |dp| <= 0.25:

| case | max abs dp before | after | max abs dlogit before | after |
|---|---|---|---|---|
| score-sequence-object-state | 1.44e-3 | **3.38e-3** | 0.435 | 0.438 |
| choice-sequence-multi-token-options | 3.8e-6 | 2.07e-3 | 0.125 | 0.063 |
| score-first-token-no-prefix | 1.05e-3 | 1.15e-3 | 0.5 | 0.75 |
| noul-sequence | 0 | 9.9e-4 | 0 | 0.374 |
| choice-first-token-image-colour | 7.0e-4 | 6.3e-4 | 0.625 | 0.375 |
| choice-first-token-image-gridworld (low margin) | 0.139, answer flipped | 0.117, answer kept | 2.38 | 2.25 |

The other cases moved 4e-4 or less. Sequence-mode readout spans of 9 to 63 rows
now take the padded int8 GEMM instead of the stock linear, which accounts for
the sequence-mode rows. `noul-sequence` (55 tokens) reached INT8 for the first
time. The worst gated case is 3.38e-3 against the 5e-3 tolerance, still well
under 1% of the smallest top-1 margin (0.934).

26B, INT8 against the stock 26B on every fixture's request
(`test_26b_int8_experts_agree_with_stock_on_fixtures_and_delta_decisions`;
|dp| 5e-3, |dlogit| 4.0, and |dp| 0.25 for the `-vt140`/`-vt70` budgets). The
largest moves were:

| case | max abs dp before | after | max abs dlogit before | after |
|---|---|---|---|---|
| choice-first-token-image-gridworld | 1.51e-3 | 3.01e-3 | 3.04 | **3.74** |
| choice-sequence-multi-token-options | 4.9e-4 | 1.72e-3 | 0.25 | 0.19 |
| choice-first-token-image-gridworld-vt140 | 0.196 | **0.248** | 2.27 | 2.02 |
| choice-first-token-image-gridworld-vt70 | 0.154 | 0.122 | 3.91 | 3.66 |

Argmax and token ids agree everywhere. **Thin margins:** the vt140 frame is at
0.248 against its 0.25 budget tolerance, and the gridworld logit is at 3.74
against 4.0. The L=1 readout used to be exact stock; it now runs on per-channel
int8 requantisations of the checkpoint's 4-bit attention weights.

Gates run 2026-10-16 (same machine, lowpowermode 0, AC):
`tests/test_int8.py`, `tests/test_memory.py`, `tests/test_int8_experts.py` and
`tests/test_moe.py` all pass (25 tests), as does
`tests/test_vision_mlx.py::test_int8_on_and_off_agree_on_image_requests`.
Swapouts rose 28146922 -> 28277492 during the 26B test runs, which hold the
stock and the INT8 26B in turn.

## Follow-up (steps 3 and 4, not started)

- **Step 3**: vendor the PolarQuant 2-bit codec for stored delta entries. The
  audit puts a 26B entry at 258 MB per 1148 tokens, so the 2 GiB cap holds
  about 8. Gate on the 20-step transcript bench (>= 4x fewer store bytes,
  argmax unchanged on fixtures).
- **Step 4**: size the store cap per model (E2B entries are 18 MB, so its
  2 GiB cap never binds and the 8-entry cap does). Expose
  `--delta-cache-mb`/entries consistently on the CLI and server, and consider
  `mx.set_cache_limit`. MLX's cache holds 0.9 to 1.5 GB of prefill transients,
  which is counted in the footprint.
- If the thin 26B margins above bite, the cheapest fix is to keep the
  checkpoint's 4-bit attention projections stock for the L=1 path. That costs
  +0.62 GB on the 26B and leaves INT8 within 8% of stock memory.
