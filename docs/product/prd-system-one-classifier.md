---
type: prd
status: active
title: System One classifier on MLX-LM — typed, calibrated decisions in one forward pass
owner: maddie
updated: 2026-10-16
---

# PRD — System One classifier on MLX-LM

## Problem

Agent harnesses, LLM-as-judge pipelines, and real-time controllers (a robot, a
game loop) need many small *decisions*: route this ticket, is this response
safe, which of four actions next. Running a chat LLM in generate mode for each
one wastes the whole decode loop producing JSON we already know the shape of,
and gives no usable confidence. TypeSafe's Jev sells a closed, hosted model for
exactly this. The mechanism is not secret: declare the options, run prefill
once, read the next-token distribution at the readout position, softmax over
the option tokens only, calibrate. See
[research: Jev and open alternatives](../research/jev-and-open-alternatives-2026-10.md).

## What we build

A **golden Python implementation** on MLX-LM, serving the same weights that
already run the chatbot (one endpoint, one VRAM budget), with:

1. A Jev-compatible wire schema (ADR-0001): `noul` / `choice` / `score`
   questions over one `state`, answers with calibrated probabilities.
2. Prefill-only inference: no decode, hidden state sliced to the readout
   position before the lm_head, stable rules prefix KV-cached.
3. First-token readout over single-token option labels, with a prefix-tree
   fallback and an opt-in sequence-scoring mode (ADR-0003).
4. Temperature-scaling calibration fitted from labelled decisions, and raw
   probabilities always exposed beside calibrated ones.
5. Gemma 4 E2B INT8 as the first target model (prefill-bound, so decisions/s
   tracks prefill tok/s), with the existing Qwen 27B judge endpoint as the
   second.
6. A fixtures directory that pins token ids and probabilities so the later
   Swift + vision port (ADR-0002) is conformance-tested, not re-derived.

## Non-goals (this repo)

- Vision input, Swift, iPhone: second repo, after the golden fixtures exist.
- Training or fine-tuning a decision model. Stock instruct weights only.
- Guaranteeing *correct* answers. We guarantee answers are drawn from the
  declared options and that probabilities are calibrated on the fitting set.

## Success criteria

- A request with three questions against a 200-token state returns in one
  forward pass with `output_tokens: 0`.
- Decisions/s on Gemma 4 E2B INT8 is within a small factor of
  prefill tok/s divided by state length (measured in Epic 2).
- Calibration ECE on a held-out set is reported, not assumed.
- Every fixture case reproduces bit-for-bit across runs on the same machine.

## Epics

See [epic registry](../epics/index.md).
