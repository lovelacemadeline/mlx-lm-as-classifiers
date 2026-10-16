---
type: research
status: active
title: Recon — where the Gemma E2B INT8 kernels live, whether VLM internals were ever vendored, and what E2B prefill does that a classifier must respect
owner: maddie
updated: 2026-10-16
---

# Recon: E2B INT8 kernels, VLM internals, E2B prefill behaviour

Read-only sweep of the sibling repos by an Opus agent on 2026-10-16 (story 2.1).
Input to Epic 2, not authoritative; verify file claims against the repos.

## A. The INT8 kernels

- **Home:** `~/Documents/dev/2026/july/int8-mlx`. Loose scripts, no pyproject.
  They **monkeypatch stock PyPI mlx-lm 0.31.3 on mlx 0.32.0**; the
  gemma4-turboquant-mlx fork is not involved. The kernels use MPP
  `matmul2d` cooperative tensors, so they need the M5 and Metal 4.
- **MLPs:** `phase9_e2e_prefill.py::Int8LinearW8A8` / `patch_mlps`. Weights are
  dequantised from `gemma-4-E2B-it-qat-4bit` and re-quantised per output
  channel to symmetric int8; activations are quantised per token at runtime
  (`fused_rowquant.py`); GEMM is `phase5_int8linear.py`. Calls under 64 rows
  fall back to stock.
- **Projections:** `agent_91_wire.py::patch_projs`. **Attention:**
  `agent_91_wire.py` swaps `scaled_dot_product_attention` for the router in
  `agent_e4b_wire4.py` (the "TP-EVERYTHING" router), which imports about eight
  sibling files.
- **The ~9k tok/s figure:** `docs/status/shipping-state-2026-07-23.md` records
  8799 tok/s prefill at 4k context with the TP-EVERYTHING router, chunk 2048,
  plugged in (1.46x over stock 5920); 8069 at 16k; about 4 to 4.8k in low-power
  mode. Bench is `phase14_w4a8_wire.py::bench_prefill`, which runs
  `stream_generate(max_tokens=1)` on random tokens and reports `prompt_tps`.
  Reproduction: `docs/runbooks/benching-and-reproduction.md` §3.
- **Cleanest vendorable copy:**
  `~/Documents/dev/2026/aug/poorly-ported-gemma4-int8-optimisations/w8a8_wiring.py`
  (275 lines, self-contained: phase5 kernel + fused_rowquant + Int8LinearW8A8 +
  patch_mlps/patch_projs, including the bf16 `.astype` fix). It walks
  `model.language_model.model.layers`, which matches E2B, but was only ever run
  on the 26B. Metal shader source is embedded as strings, so it is portable
  to Swift later.

## B. Vendored VLM internals: none

No repo under `~/Documents/dev/2026` contains an MLX vision tower, SigLIP, or
image-token merging outside site-packages. The grep hits were weight-stripping
code: the fork's `gemma4.py` sanitize drops `vision_tower`, `embed_vision`,
`audio_tower`; `e4b-vs-e2b/build_configM_model.py` does the same. The only
INT8-plus-mlx-vlm wiring (`int8-mlx/agent_e7_diffusion_wire.py`, story 7.7)
targeted the diffusion 26B MoE in text-only use. **For E2B it is mlx-lm only,
text only, and vision has never been exercised with these kernels.**

## C. Known gaps in the INT8 path

- Text-only top-1 fidelity vs stock: 0.981 mean, 0.953 worst stratum
  (int8-mlx story 9.4).
- **The attention router only engages when prompt length is a multiple of
  64**; other lengths fall back to stock attention. The fix (story 9.3,
  "L-padding") is still pending there. Pad with a Q-side mask, not with filler
  text, which would change the classifier input.
- The router assumes batch size 1 and ignores `mask` (causal only).
- The runbook's champion router is stale (`agent_dbg_wire_fixed`); the kernel
  inventory says TP-EVERYTHING superseded it.

## D. E2B prefill behaviour the classifier must respect

- **KV sharing is the big one.** `num_kv_shared_layers=20`: of 35 layers,
  only 0 to 14 produce K/V; 15 to 34 reuse it. mlx-lm's cache-only prefill
  therefore runs **only the 15 producer layers**, which is where the headline
  prefill speed comes from (`e4b-vs-e2b/FINDINGS.md`, citing `gemma4_text.py`
  and `generate.py`). A logit readout **must run layers 15 to 34**. Done naively
  over every row that is roughly 3x the FLOPs (producers are about a third of
  the full forward).
- **The fix:** run layers 0 to 14 over all tokens, then run 15 to 34 on **the
  readout row only**. Consumer layers read the shared K/V plus their own
  residual, so this is exact. Slice `per_layer_inputs` the same way. MLX's lazy
  evaluation will not prune rows on its own.
- Sliding layers are D=256 with window 512; globals are D=512; only about
  three globals execute. KV-shared layers have no `k_proj`/`v_proj`.
- `adaptive_prefill.py` layer skipping targets E4B and is experimental.
  "config-M" is the 26B MoE distillation. Both are out of scope.

## E. Recommendation for Epic 2

1. Pin `mlx==0.32.0`, `mlx-lm==0.31.3` from PyPI. Do not depend on the fork.
2. Vendor `w8a8_wiring.py` for MLP and projection INT8 (about 1.3 to 1.4x).
   Attention router second, behind a flag, with the L-padding done properly.
3. Build the readout-row consumer pass (story 2.5) before worrying about
   kernels: it is the larger win for a classifier and it is model-structural.
4. Reuse `int8-mlx/agent_e2b_stratified_eval.py` as the int8-vs-stock parity
   gate; `phase14_w4a8_wire.py::logits_probe` is the nearest thing to a
   readout and runs the full forward unsliced.

## F. Nothing existing does option-masked readout or calibration

Closest: `logits_probe` above, `e4b-vs-e2b/lens_check.py` (logit lens),
`capture_teacher_kd.py` (top-k log_softmax capture), and
`aug/kv-concat/kv_stitch.py` (stitching independently prefilled KV chunks,
possibly useful for the shared-prefix cache later).
