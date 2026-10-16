---
type: story
status: completed
title: MoE expert INT8 for the 26B — vendor the W4A8 expert gather-GEMM (ling3-mlx kernel, int8-mlx affine fold)
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/int8/experts.py::Int8SwitchGLU"
  - symbol: "src/mlx_classifiers/int8/experts.py::Int8SwitchLinear"
  - symbol: "src/mlx_classifiers/int8/experts.py::gather_gemm"
  - symbol: "src/mlx_classifiers/int8/experts.py::quant_rows"
  - symbol: "src/mlx_classifiers/int8/experts.py::build_tiles"
  - symbol: "src/mlx_classifiers/int8/experts.py::patch_experts"
  - symbol: "src/mlx_classifiers/mlx_runner.py::MLXRunner.load"
  - test: "tests/test_int8_experts.py::test_quant_rows_matches_the_reference"
  - test: "tests/test_int8_experts.py::test_gather_gemm_matches_int8_times_dequantised_weights"
  - test: "tests/test_int8_experts.py::test_patch_experts_swaps_switch_glu_and_stays_close_at_every_row_count"
  - test: "tests/test_int8_experts.py::test_26b_int8_experts_agree_with_stock_on_fixtures_and_delta_decisions"
---

# Story 2.8 — MoE expert INT8

`--int8` gave the 26B MoE about 5% (`bench_runs/bench-20261016-182238`
vs `-182057`). The vendored W8A8 wiring (story 2.3) patches the dense MLPs
and attention projections and leaves the experts (`Experts.switch_glu`, a
`SwitchGLU` in mlx-lm 0.31.3) stock. With `int8=True` the experts now run a
W4A8 gather-GEMM too. The 26B prefills 1.26x to 1.43x faster uncached, and
delta decisions run 1.21x to 1.29x faster.

## Provenance: where the ~1k tok/s figure came from

The brief named `aug/poorly-ported-gemma4-int8-optimisations` as the source.
**That repo has no expert INT8 path.** Its benchmark (`bench_gemma4_prefill.py`)
is W8A8 dense and attention (`w8a8_wiring.py`, already vendored by 2.3)
with **stock 4-bit experts and top-2 routing** (`router_patched`, `MODE =
topk k=2`). It measured 451.8 tok/s, then 473.7 to 501.1 after its story 1.1.
Its story 1.2 inventories the expert kernels and concludes there was "nothing
beyond what's already ported worth porting".

The ~1k tok/s 26B figure is int8-mlx story 5.17
(`july/int8-mlx/docs/stories/5-17-w4a8-gather-and-interleaved-rebench.md`,
harness `agent_e5_w4a8_final.py`, log `agent_e5_w4a8_final.log`, git
`f924f0f` for the kernel). That lane was **k2+W4A8 @ chunk 1024: 1054.4 tok/s**
(median of 5 interleaved rounds). The conditions were: 4096 random tokens,
lowpowermode 0, AC power, and the W8A8 dense and attention wiring on.
**Prefill routing was cut to top-2 experts**, which changes the model; its
fidelity was 0.938 against the 0.95 bar. The kernel was
`agent_e5_affine_gather.py`, a native-affine sorted gather wired per
projection by `agent_e5_chunksweep.py::e5_call`. 1054.4 tok/s was 0.72x of
the stock-expert lane at the same k2 (1455.6). Story 5.21 traced the
deficit to fp16 output promoting the bf16 stream to fp32, and one bf16 cast
brought it to parity with stock experts (-9 ms on 3128). So the ~1k figure
owes most of its speed to top-2 routing and the dense wiring, not to the
expert kernel. Top-2 routing is not vendored: a classifier must not change
the model.

The expert kernel with a measured win is in `aug/ling3-mlx/moe_gather_int8.py`
(git `2b97d6f`, ling3 epic 2, stories 2.1 to 2.3). It is a W4A8 gather over
the checkpoint's nibbles read in place: XOR to `int4b_format` with
`transpose_right`, 16-row tiles, and a device-side tile table. It measured
1.25x to 1.30x at T=512 and 1475 -> 1623 tok/s served at 2048 on Ling-3.
It assumes symmetric int4 with group 32.

**The 26B's experts are affine int4 with group 64.** 34% of the groups in
layer 0 have `b != -8s`; the largest `|b + 8s|` is 0.049 for up and 0.107 for
down. So the ling3 kernel alone is wrong for this model.

## What was vendored

`src/mlx_classifiers/int8/experts.py` has a provenance header with both
sources, their git revs and their sha256 digests. It is a merge of the two,
not a byte copy:

- **The kernel** is ling3's `int8 x int4b` `matmul2d`: SM=16, SN=64, SK=32,
  `transpose_right`, and the XOR repack, so the expert bytes are reused at the
  same size. It carries **int8-mlx 5.3/5.17's affine fold**, applied once per
  64-wide group (two K steps): `acc += s * int(xq . (u-8)) + (b + 8s) *
  rowsum_g(xq)`. This is exact against `mx.dequantize`, so the weights are used
  bit-exact. The output is bf16 (the 5.21 fix).
- **`build_tiles`** comes from ling3, with its zero pad tile. The per-expert
  counts are now a scatter-add instead of a rows x experts comparison.
- **`quant_rows`** is new. It is one fused Metal pass: 2.3's per-row quantiser
  plus the int32 per-group code sums. It is bit-equal to
  `_quant_rows_reference`. Unfused mx ops made the whole layer 2x slower than
  stock at 636 tokens and more.
- **`Int8SwitchGLU`** quantises once per token before the gather, as ling3
  does. It runs up and gate, GeGLU, requantises, then runs down.
  `patch_experts` converts each eligible `SwitchGLU` (4-bit affine, group 64,
  no bias) and **drops the stock expert arrays**. Scales and `b + 8s` are kept
  as fp32 `(E, K/64, N)`.

The Router (`router.proj`, 8-bit, 128 outputs) stays stock; it is a few
microseconds per layer.

**M5 only.** The kernel needs MPP `matmul2d` cooperative tensors with
`int4b_format` (Metal 4). The tests skip unless `moe.available()` (MLX 0.32 on
an M5 GPU). The kernel compiled and matched on the first attempt.

### Coexistence with moe.py (story 3.6): replace, not dispatch

The 3.6 kernel reads the stock nibble encoding, and memory rules out keeping
both copies: 11.4 GB of expert weights on a 32 GB machine. So under
`int8=True` on a MoE, the W4A8 path **replaces** `SegmentedSwitchGLU` at every
row count, including the L=1 readout. `MLXRunner.load` runs `patch_experts`
before `moe.patch_model`, which then finds no stock `SwitchGLU` and swaps 0
layers. `int8_experts=False` keeps the stock experts with the 3.6 kernel under
`int8`.

The replacement is at least as fast where 3.6 ran. In the delta cells
(102-token prefills, inside 3.6's window) p50 drops from 222.9-255.8 ms
to 173.5-211.4 ms. On a toy 26B-shaped layer (`SwitchGLU` 2816 -> 704,
128 experts, top-8, random routing) the times were:

| L | stock ms | 3.6 kernel ms | W4A8 ms |
|---|---|---|---|
| 1 | 0.42 | 0.42 | 1.08 |
| 97 | 14.48 | 14.96 | 5.01 |
| 256 | 18.29 | 17.72 | 5.97 |
| 636 | 26.94 | 26.71 | 9.40 |
| 1148 | 35.46 | 35.13 | 19.10 |
| 2048 | 61.54 | 60.36 | 33.05 |

The cost is at **L=1**: about 0.66 ms of fixed overhead per layer (argsort,
quantise, tile table, three launches), so about 20 ms on the 30-layer readout
pass. The delta cells already include that cost.

**Memory:** MLX active memory is 14.47 GB stock, 16.12 GB with dense INT8
only, and 17.54 GB with INT8 experts. The fp32 scales and `b + 8s` add
+1.4 GB over the bf16 originals. Load takes 8.2 s, against 3.6 s stock.

## Parity

Tolerance: **raw probabilities within 5e-3** (`tests/test_int8_experts.py::PROB_TOL`,
the E2B gate of 2.3). Argmax and token ids must match exactly. Raw logits
must be within **3.0**, a 26B-specific `LOGIT_TOL`. E2B's 1.0 does not hold
on this model even for 2.3's dense-only INT8, which moves the 26B's logits
by up to 1.75.

The committed fixtures were generated on E2B, so their `expected` blocks are
E2B logits. The 26B test therefore runs every committed fixture's request
through the stock 26B, uses those results as the expectation, then runs them
with `int8=True`. It also runs three delta-cached decisions at 512 tokens with
a 50-token tail. Measured on 2026-10-16 (M5, lowpowermode 0, battery):

| case | max abs dp | max abs dlogit |
|---|---|---|
| choice-first-token-cached-list-state | 5.7e-10 | 0.375 |
| choice-first-token-multi-token-label-error | (errors, as expected) | - |
| choice-first-token-object-state | 6.0e-8 | 0.438 |
| choice-sequence-multi-token-options | **4.9e-4** | 0.25 |
| noul-first-token | 1.3e-8 | 1.19 |
| noul-sequence | 6.1e-9 | 0.094 |
| score-first-token-no-prefix | 6.5e-9 | 1.12 |
| score-sequence-object-state | 4.8e-7 | **2.33** (an option at p 1.5e-8) |
| bench prompts at 512 and 1024 tokens, 3 delta decisions | <= 1.7e-7 | - |

Argmax agrees on every case. `tests/test_moe.py::test_26b_delta_decisions_are_identical_with_the_kernel_on_and_off`
(int8 off) still passes. The E2B INT8 fixture test (`tests/test_int8.py`)
also passes: E2B is dense, so `patch_experts` swaps 0 layers there.

## Bench (2026-10-16, M5 MBA, lowpowermode 0, battery)

`mlx-classifiers bench --model mlx-community/gemma-4-26B-A4B-it-qat-4bit
--lengths 128 512 1024 --reps 5 --delta-tail 50`, without and then with
`--int8`, run back to back. mlx 0.32.0, mlx-lm 0.31.3, git `722aae5-dirty`,
GPU free. Power before, between and after the runs was `lowpowermode 0` on
battery: "Now drawing from 'Battery Power'", 94% -> 93%, discharging.

INT8 off (`bench_runs/bench-20261016-192915`; the 3.6 kernel on, 30 layers):

| state tok | prompt tok | prefix cached | delta tail | prefilled tok | cached tok | p50 ms | p95 ms | decisions/s | prefill tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 252 | no | - | 252 | 0 | 363.2 | 363.8 | 2.8 | 694 |
| 128 | 252 | yes | - | 186 | 66 | 290.1 | 291.8 | 3.4 | 641 |
| 128 | 252 | yes | 50 | 102 | 150 | 222.9 | 224.3 | 4.5 | 458 |
| 512 | 636 | no | - | 636 | 0 | 720.0 | 722.1 | 1.4 | 883 |
| 512 | 636 | yes | - | 570 | 66 | 669.5 | 672.5 | 1.5 | 851 |
| 512 | 636 | yes | 50 | 102 | 534 | 234.5 | 234.6 | 4.3 | 435 |
| 1024 | 1148 | no | - | 1148 | 0 | 1143.5 | 1157.7 | 0.9 | 1004 |
| 1024 | 1148 | yes | - | 1082 | 66 | 1087.7 | 1090.6 | 0.9 | 995 |
| 1024 | 1148 | yes | 50 | 102 | 1046 | 255.8 | 258.0 | 3.9 | 399 |

INT8 on (`bench_runs/bench-20261016-192959`; dense W8A8 plus expert W4A8 on
30 layers, 3.6 kernel on 0 layers):

| state tok | prompt tok | prefix cached | delta tail | prefilled tok | cached tok | p50 ms | p95 ms | decisions/s | prefill tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 252 | no | - | 252 | 0 | 250.3 | 250.5 | 4.0 | 1007 |
| 128 | 252 | yes | - | 186 | 66 | 212.1 | 212.7 | 4.7 | 877 |
| 128 | 252 | yes | 50 | 102 | 150 | 173.5 | 173.6 | 5.8 | 588 |
| 512 | 636 | no | - | 636 | 0 | 502.9 | 504.0 | 2.0 | 1265 |
| 512 | 636 | yes | - | 570 | 66 | 465.0 | 466.0 | 2.2 | 1226 |
| 512 | 636 | yes | 50 | 102 | 534 | 186.3 | 188.8 | 5.4 | 548 |
| 1024 | 1148 | no | - | 1148 | 0 | 907.9 | 913.8 | 1.1 | 1264 |
| 1024 | 1148 | yes | - | 1082 | 66 | 882.7 | 885.5 | 1.1 | 1226 |
| 1024 | 1148 | yes | 50 | 102 | 1046 | 211.4 | 223.0 | 4.7 | 482 |

Uncached prefill gains 1.45x at 128 state tokens, 1.43x at 512 and 1.26x at
1024. Delta decisions gain 1.21x to 1.29x. These numbers were taken with low
power mode off, so they do not compare with 3.6's low-power-mode tables.

A rough split from the parity probe (one decision each, same session): the
1024-token decision took 0.72 s stock, 0.65 s with dense INT8 only, and
0.50 s with expert INT8 as well. Most of the gain is from the experts.

## Acceptance Criteria
- **Given** the 26B with `int8=True` **when** the fixtures run **then** argmax
  agrees and probabilities are within the stated tolerance. **Met**: the
  tolerance is 5e-3 on probabilities, with argmax and token ids exact, against
  the stock 26B on every committed fixture's request and on three delta
  decisions. The worst case measured was 4.9e-4. Logits are within 3.0, the
  worst measured 2.33
  (`test_26b_int8_experts_agree_with_stock_on_fixtures_and_delta_decisions`).
- **Given** the bench at 512 and 1024 tokens, uncached **when** INT8 is on
  **then** prefill tok/s improves materially over stock, recorded with power
  state. **Met**: 883 -> 1265 tok/s (1.43x) at 512 and 1004 -> 1264 tok/s
  (1.26x) at 1024, on battery with lowpowermode 0.

## Notes and limits
- The L=1 path carries about 0.66 ms per layer of fixed overhead over stock.
  A stock-numerics small-row path would need the stock encoding back: either
  XOR back the touched experts per call, or keep both copies. Neither was
  worth it here, because the delta decisions are net faster.
- The memory cost is +1.4 GB for the fp32 fold operands, against
  int8-mlx's +2.7 GB. bf16 `b + 8s` would halve it but round the fold.
- `patch_experts` only takes 4-bit affine experts with group 64. Any other
  checkpoint keeps stock experts.
