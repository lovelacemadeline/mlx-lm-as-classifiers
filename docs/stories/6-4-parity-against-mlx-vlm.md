---
type: story
status: completed
title: Parity test against mlx-vlm as a dev-only reference
owner: maddie
updated: 2026-10-16
anchor:
  - test: "tests/test_vision_parity.py::test_merged_embeddings_and_readout_match_mlx_vlm"
  - symbol: "tests/vlm_reference.py::reference"
---

# Story 6.4 — Parity test against mlx-vlm as a dev-only reference

`tests/test_vision_parity.py` sends the same PNG and the same prompt token ids
(our compiler's) through the vendored path and through mlx-vlm 0.7.8, on E2B
and on the 26B-A4B (each skipped when its weights are not cached).

**Where mlx-vlm comes from.** The `vision-dev` extra (`mlx-vlm>=0.7.8`)
installs it in process. But mlx-vlm 0.7.8 requires `mlx>=0.32.3`, and this
repo's environment is pinned at mlx 0.32.0 (Epic 2), so installing the extra
here would move the pin under the INT8 kernels and the fixtures. The test
therefore also accepts `MLX_VLM_PYTHON=<interpreter with mlx-vlm>` and runs
`tests/vlm_reference.py` there as a subprocess (`-I`), exchanging token ids,
the PNG and an `.npz`. With neither, the test is skipped. It runs the
reference first and exits it, so the two 26B copies never coexist (32 GB
machine). This is a deviation from "an extra" alone, kept deliberately.

**What is compared, and the result** (2026-10-16, M5, low power mode off;
ours mlx 0.32.0, reference mlx-vlm 0.7.8 on mlx 0.32.4):

| quantity | E2B | 26B-A4B | gate |
|---|---|---|---|
| pixel values (`[1, 3, 864, 720]`) | exact | exact | exact |
| projected soft tokens (`[270, 1536]` / `[270, 2816]`) | exact | exact | exact |
| merged input embeddings, max abs diff | 0 | 0 | `EMBED_TOL` 0.0625 |
| per-layer inputs, max abs diff | 0 | n/a (none on the 26B) | `EMBED_TOL` |
| option-label logits at the readout, max abs diff | 0 | 0.0625 | `LOGIT_TOL` 0.2, same argmax |

E2B is bit-identical end to end: our split prefill on merged embeddings
gives mlx-vlm's full-forward logits exactly at the option ids. On the 26B the
embeddings are exact and the readout differs by one bf16 step (our MoE kernel
and split prefill against mlx-vlm's full forward, on different mlx versions).
The tolerances are stated as a bf16 step at the largest magnitude (embeddings)
and 0.2 for logits, which is tight enough to catch the one 26B-specific piece:
run without the bidirectional image-block mask, the same 26B case moves by
0.31 (11.875 / -5.406 / -8.625 / -8.438 against -5.5 / -8.5 / -8.125 with it).

## Acceptance Criteria
- **Given** mlx-vlm available (in process or via `MLX_VLM_PYTHON`) **when** the parity test runs **then** pixel values and soft tokens are exact, merged embeddings and per-layer inputs are within `EMBED_TOL` (measured 0), and the readout's option logits agree in argmax and within `LOGIT_TOL` (measured 0 on E2B, 0.0625 on the 26B). *(test_merged_embeddings_and_readout_match_mlx_vlm)*
- **Given** mlx-vlm absent **when** the suite runs **then** the test is skipped and nothing else needs it. *(same test, skip path)*
