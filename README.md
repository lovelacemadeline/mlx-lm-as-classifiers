---
type: guide
status: active
title: mlx-lm-as-classifiers — README
owner: maddie
updated: 2026-10-16
---

# mlx-lm-as-classifiers

Typed, calibrated decisions from an MLX-LM decoder in one forward pass. Declare
the questions (`noul` yes/no, `choice` enum, `score` ordered scale), pass the
program state, get probabilities back. No decode loop: the logits at the last
prefill position are the decision.

This is the **golden Python implementation**. A Swift + vision port for iPhone
lives in a separate repo and must reproduce the fixtures under `fixtures/`.

Docs follow [`docs/doc-conventions.md`](docs/doc-conventions.md) and are
checked by `docloom check`. Start at [`docs/product/prd-system-one-classifier.md`](docs/product/prd-system-one-classifier.md).
