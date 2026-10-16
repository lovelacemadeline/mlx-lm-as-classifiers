---
type: story
status: in-progress
title: KV-consumer layers run on the readout row only
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "proto/readout_cost.py::split"
  - symbol: "proto/readout_cost.py::naive"
  - symbol: "proto/floor.py::last_full"
---

# Story 2.5 — KV-consumer layers run on the readout row only

E2B shares K/V from layers 0 to 14 into layers 15 to 34. Cache-only prefill
runs only the producers, which is where the fast prefill number comes from, but
a logit readout needs the consumers. Run producers over all tokens, then run
consumers over the readout row(s) only, slicing the residual and
`per_layer_inputs` to those rows. Exact, because consumers read shared K/V plus
their own residual. See the
[recon](../research/e2b-int8-and-vlm-recon-2026-10.md) §D.

## Acceptance Criteria
- **Given** a prompt **when** the sliced path and the full forward both run
  **then** readout logits match to within bf16 noise on every fixture.
- **Given** a 2k-token prompt **when** profiled **then** consumer layers see
  only the readout rows (test by asserting the hidden-state shape entering
  layer 15).
- **Given** the sliced path **when** benchmarked (story 2.4) **then** decisions/s
  approaches cache-only prefill tok/s divided by prompt length, not a third
  of it.

## Findings (2026-10-16, prototype, M5 MBA, **low power mode on**, AC)

The split path already exists in mlx-lm: `generate_step` prefills all but the
last token evaluating only the cache state, and MLX's lazy evaluation prunes
the 20 consumer layers and the lm_head for that chunk. The last token then runs
all 35 layers at L=1 (a decode step). `proto/readout_cost.py` measures naive
(full forward, slice last row) against split, random tokens, best of three:

| N tokens | naive ms | split ms | ratio | split tok/s | argmax agree |
|---|---|---|---|---|---|
| 128 | 258 | 116 | 2.2 | 1100 | yes |
| 512 | 600 | 199 | 3.0 | 2570 | yes |
| 1024 | 1303 | 351 | 3.7 | 2915 | yes |
| 2048 | 2515 | 774 | 3.3 | 2650 | yes |

So the recon's "3x" is real and the fix is free for one readout per sequence:
use the generate-style split, never `model(all)[:, -1]`.

`proto/floor.py` at N=256: cache-only prefill of 255 tokens 70 ms; the last
token through all layers 30 ms, of which the lm_head is under 3 ms. **The
per-decision floor is the L=1 pass through 35 layers, not the vocab
projection.** With a 256-token prefix cached, a decision is roughly 47 ms end
to end here, so about 20 per second in low power mode before any INT8 work.

Caveats and follow-ups:
- Max absolute logit difference between naive and split is 0.4 to 0.75 (bf16
  plus different matmul shapes). Argmax agrees at every N. Fixtures (Epic 4)
  must pin the split path, not compare across paths.
- Multiple readout rows per sequence (packed questions) are not covered by the
  split: consumer layers would have to run over gathered rows with a per-row
  causal mask against the cache. That is the remaining scope of this story;
  the single-readout case is solved by construction.
- Re-measure plugged in with low power mode off before quoting numbers.
