---
type: story
status: completed
title: Delta cache fixed overhead — the 600 ms floor on the 26B
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/moe.py::segmented_gather_qmm"
  - symbol: "src/mlx_classifiers/moe.py::SegmentedSwitchGLU"
  - symbol: "src/mlx_classifiers/moe.py::patch_model"
  - symbol: "src/mlx_classifiers/moe.py::use_segmented"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.load"
  - test: "tests/test_moe.py::test_use_segmented_window"
  - test: "tests/test_moe.py::test_segmented_gather_qmm_is_bit_identical_to_gather_qmm"
  - test: "tests/test_moe.py::test_patch_model_swaps_quantized_switch_glu_and_keeps_outputs"
  - test: "tests/test_moe.py::test_26b_delta_decisions_are_identical_with_the_kernel_on_and_off"
---

# Story 3.6 — Delta cache fixed overhead

Bench on 2026-10-16 (low power mode, `bench_runs/bench-20261016-182057`):
with delta caching the 26B MoE prefills only 102 tokens per decision at every
state length, yet p50 stays at 670 to 720 ms, so decisions/s plateaus at
about 1.5. E2B's equivalent floor is about 100 ms. The suspects were deep
copies of the stored KV entry on lookup and store (the 26B's entry is
246 MiB at 1046 tokens), the L=1 readout pass through the MoE, and the
store's byte accounting.

**The suspects were wrong.** The bookkeeping costs under 2 ms per decision.
The floor is the delta's own ~100-token prefill through the 26B's experts:
MLX's expert matmul kernel pads each expert's handful of rows to a 64-row
tile. A per-expert kernel (`mlx_classifiers/moe.py`) now runs those rows in
16-row tiles. Its output is bit-identical to the stock path. The 1024-token
delta decision on the 26B drops from 778 to 547 ms (1.3 to 1.8
decisions/s). That is 30%, not the 50% the second criterion asks for. The
rest of the cost is below.

## Profile

`proto/delta_floor.py` builds the bench's delta-tail prompts (1024-token
state, last 50 tokens changing, hit 1046 tokens), warms the store with real
decisions, then replays `MLXRunner.prefill` phase by phase with
`mx.synchronize()` fences: median of 7 decisions. A whole
`runner.prefill` + answer is timed alongside each one as a check on the sum.
26B: `gemma-4-26B-A4B-it-qat-4bit`. E2B: `gemma-4-E2B-it-qat-4bit`. M5 MBA,
low power mode on, AC power, GPU free.

| phase (one decision, 1024 state, 50-token tail) | 26B before | share | 26B after | E2B |
|---|---|---|---|---|
| lookup: candidate scan + deepcopy + trim | 2.2 ms | 0.3% | 1.2 ms | 0.9 ms |
| cache-only prefill of the delta (97 tokens) | **671.9 ms** | **93%** | 476.5 ms | 64.4 ms |
| store: deepcopy of the snapshot | 0.5 ms | 0.1% | 0.4 ms | 0.2 ms |
| store: `cache_nbytes` | 0.0 ms | 0% | 0.0 ms | 0.0 ms |
| store: insert + LRU eviction | 0.1 ms | 0% | 0.1 ms | 0.1 ms |
| readout: L=1 backbone pass | 42.5 ms | 6% | 43.6 ms | 28.7 ms |
| readout: head + logsumexp | 5.3 ms | 0.7% | 5.2 ms | 2.7 ms |
| answer (masked softmax) | 1.1 ms | 0.2% | 1.0 ms | 0.8 ms |
| sum of phases | 723.5 ms | | 528.2 ms | 97.9 ms |
| whole decision, timed separately | 719.4 ms | 100% | 531.4 ms | 98.1 ms |

Deep copies and byte accounting are noise. `copy.deepcopy` of a 246 MiB
cache list takes 0.4 ms, so it cannot be copying the buffers: at the
measured 130 GB/s that alone would take about 4 ms. Move semantics,
in-place trims, store-only-when-new and O(1) accounting would each save
under a millisecond, so none was made.

Inside the cache-only prefill (`proto/delta_sweep.py`, `proto/moe_share.py`,
random tokens on a 1046-token cache):

- **Cost against delta length L** is the expert-coverage curve. L=1: 32 ms;
  8: 147; 16: 246; 32: 373; 64: 466; 97: 545; 128: 594; 256: 830 ms. Top-8
  of 128 experts touches every expert from about L=40. Past that the cost
  grows by about 1.9 ms per token.
- **Experts' share.** Replacing `SwitchGLU` with a zeros stub takes the
  97-token prefill from 548 to 159 ms. So 389 ms (71%) is the expert
  matmuls, plus their sort, gather and scatter. At L=1 the same stub saves
  2.7 ms (8%).
- **Why.** With L=97 the sorted gather has 776 (token, expert) rows, about 6
  per expert. MLX 0.32 runs them through `affine_gather_qmm_rhs_nax` (Apple
  M5) in 64-row tiles. A tile that spans k experts runs k full 64-row
  matmuls (the `while (n < tgp_bm)` loop over index runs). So the kernel
  does about 140 matmuls of 64 rows each for 776 useful rows, roughly 10x
  the work. MLX takes this path only from `rows >= 4 * experts`. Below that
  it uses a per-row kernel with different rounding.

## Fix: a per-expert gather-QMM

`segmented_gather_qmm` gives each threadgroup one expert (grid y) and one
64-column block (grid x). It runs 16-row tiles over that expert's rows only,
from `expert_offsets` (a count and cumsum over the sorted ids). An expert
with no rows exits at once. The tile body is MLX's own `qmm_t_nax_tgp_impl`:
MLX's `QuantizedBlockLoader` dequantises the weights, the matmul is
`tile_matmad_nax`, and K is walked in the same order. The output is
therefore **bit-identical** to `mx.gather_qmm`. The header comes from the
`quantized_nax.h` and `steel/gemm/nax.h` that ship inside the installed mlx
wheel, so the code is not copied into this repo.

`SegmentedSwitchGLU` uses the kernel only when
`4 * experts <= rows <= 16 * experts`. Below that window MLX's own path has
different numerics. Above it the stock 64-row tiles are no longer mostly
padding. On the 26B the window is 64 to 256 tokens. Everything else,
including the L=1 readout pass, runs the stock code.

`MLXRunner.load` calls `moe.patch_model`, which swaps the class of each
eligible `SwitchGLU` (4-bit affine, group 64, no bias). It does nothing on
dense models (E2B: 0 layers), on any mlx other than 0.32, or on a GPU
without NAX. It also refuses if a one-off bit-identity check against
`mx.gather_qmm` fails. Turn it off with `moe_kernel=False` or
`bench --no-moe-kernel`. The bench header records the number of layers
swapped.

Measured (`proto/moe_nax.py`; `proto/moe_nax_model.py` in the model):

- One 26B-shaped expert projection (128 x 704 x 2816), 20 calls back to
  back, L=97: stock 7.5 ms, per-expert 4.0 to 4.3 ms, bit-equal.
- 26B cache-only prefill on a 1046-token cache, before and after, with the
  next token's hidden state bit-identical in every case:

| delta L | before (ms) | after (ms) | speedup |
|---|---|---|---|
| 64 | 458.6 | 368.3 | 1.25x |
| 97 | 546.1 | 419.1 | 1.30x |
| 102 | 555.0 | 431.7 | 1.29x |
| 128 | 610.3 | 478.6 | 1.28x |
| 256 | 830.7 | 775.7 | 1.07x |

Tried without gain (`proto/moe_nax_db.py`, `proto/moe_nax3.py`):

- Double-buffered weight blocks: no change.
- One weight row per thread with 16-byte loads and a register prefetch:
  slower.
- Tile shapes BN 32, 64 or 128 with BK 32 or 64: all within noise.
- A scalar multi-row QMV: 2 to 4x slower than stock.

`proto/moe_nax_probe2.py` splits the cost of one projection at L=97: the
weight loader alone takes 1.9 ms, the NAX matmuls alone 0.7 ms, and the full
kernel 4.3 ms. The two run one after the other, not overlapped. Removing
that serialisation is the next lever for the expert matmuls.

## Acceptance Criteria
- **Given** the profile **when** recorded in this story **then** each suspect
  has a measured share of the 26B's per-decision wall time. **Met** (table
  above). Lookup copy 0.3%, store copy 0.1%, byte accounting 0%, L=1 readout
  6.6% with its head, and the delta's cache-only prefill 93%, of which the
  expert matmuls are about 71%.
- **Given** the fix **when** the delta bench reruns on the 26B at 1024 tokens
  **then** p50 drops by at least half with identical probabilities.
  **Partly met.** p50 went from 777.9 to 546.6 ms, a 30% drop (1.3 to 1.8
  decisions/s), with probabilities identical bit for bit:
  `test_26b_delta_decisions_are_identical_with_the_kernel_on_and_off`
  compares the delta decisions' `raw_probabilities` with the kernel on and
  off, and `test_segmented_gather_qmm_is_bit_identical_to_gather_qmm` /
  `test_patch_model_swaps_quantized_switch_glu_and_keeps_outputs` check it
  on synthetic experts. The E2B fixtures, the 3.4 delta-cache tests and the
  rest of `pytest -m mlx` pass (21). Halving needs the decision at about
  390 ms. What is left of 547 ms: about 300 ms of expert matmuls (loader
  bound, see above), about 160 ms of non-expert layers over the 97-token
  delta (measured with the expert stub on random tokens), and the 44 ms
  readout pass.
  [2.8](2-8-moe-expert-int8.md) (expert INT8) is where the expert GEMM goes
  next.

## Bench (2026-10-16, M5 MBA, low power mode on)

`mlx-classifiers bench --model mlx-community/gemma-4-26B-A4B-it-qat-4bit
--lengths 128 512 1024 --reps 5 --delta-tail 50`, mlx 0.32.0, mlx-lm 0.31.3,
git `cfa0115-dirty`, AC power, GPU free. "Before" is the same build with
`--no-moe-kernel` (`bench_runs/bench-20261016-185450`), run back to back
with "after" (`-185623`). The original run (`-182057`, git `f1591f5`) had
the 1024-token delta cell at 722.9 ms.

| state tok | prefix cached | delta tail | prefilled tok | cached tok | before p50 ms | before dec/s | after p50 ms | after dec/s |
|---|---|---|---|---|---|---|---|---|
| 128 | no | - | 252 | 0 | 931.5 | 1.1 | 799.2 | 1.3 |
| 128 | yes | - | 186 | 66 | 795.3 | 1.3 | 642.6 | 1.6 |
| 128 | yes | 50 | 102 | 150 | 669.0 | 1.5 | 487.7 | 2.1 |
| 512 | no | - | 636 | 0 | 1646.1 | 0.6 | 1685.4 | 0.6 |
| 512 | yes | - | 570 | 66 | 1519.5 | 0.7 | 1530.1 | 0.7 |
| 512 | yes | 50 | 102 | 534 | 692.9 | 1.4 | 511.1 | 2.0 |
| 1024 | no | - | 1148 | 0 | 2629.4 | 0.4 | 2667.8 | 0.4 |
| 1024 | yes | - | 1082 | 66 | 2723.3 | 0.4 | 2616.7 | 0.4 |
| 1024 | yes | 50 | 102 | 1046 | 777.9 | 1.3 | 546.6 | 1.8 |

Prompts of 512 tokens and more sit outside the kernel's window, so those
rows run the stock path and differ only by noise. The 252-token uncached
prompt (2016 rows) is inside the window and gains 14%.

E2B, same command without `--model` (`bench_runs/bench-20261016-185644`).
The laptop had **switched to battery power** by this run (low power mode
still on). E2B is dense, so the kernel swaps 0 layers and its path is
unchanged.

| state tok | prefix cached | delta tail | prefilled tok | cached tok | p50 ms | decisions/s |
|---|---|---|---|---|---|---|
| 128 | no | - | 248 | 0 | 108.5 | 9.2 |
| 128 | yes | - | 182 | 66 | 89.9 | 11.1 |
| 128 | yes | 50 | 98 | 150 | 95.2 | 10.5 |
| 512 | no | - | 632 | 0 | 206.9 | 4.8 |
| 512 | yes | - | 566 | 66 | 191.8 | 5.2 |
| 512 | yes | 50 | 98 | 534 | 97.4 | 10.3 |
| 1024 | no | - | 1144 | 0 | 358.3 | 2.8 |
| 1024 | yes | - | 1078 | 66 | 342.7 | 2.9 |
| 1024 | yes | 50 | 98 | 1046 | 98.0 | 10.2 |

E2B's delta floor is flat at about 97 ms from 128 to 1024 tokens. The 190.8
ms that 3.4 recorded at 1024 does not reproduce, and its "overhead that
still grows with the cached length" is not visible here.

## Notes and limits
- The kernel builds its header from the mlx wheel's installed headers.
  `available()` pins mlx 0.32.x. A new mlx needs this re-checked (and the
  self-check will refuse a mismatch) before the pin moves.
- Bit-identity holds against MLX's NAX path only, which is why the window
  starts at 4 rows per expert. The kernel never runs where stock numerics
  come from a different kernel.
- INT8 (`--int8`, story 2.3) patches dense MLPs and projections, not
  `SwitchGLU`, so the two compose.
