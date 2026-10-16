---
type: decision
status: active
title: Golden Python MLX-LM implementation first; Swift + vision port in a second repo, conformance-tested by fixtures
owner: maddie
updated: 2026-10-16
---

# ADR-0002 — Golden Python first, Swift second

## Status
Active.

## Context
The end goal includes an iPhone (Swift, MLX Swift, vision input) build that is
vendored into app stacks (first: an EV3 servo controller). Doing that first
means debugging tokeniser, readout-position, bias and calibration issues on a
phone, with vision in the loop.

## Decision
1. **This repo is the golden implementation**: Python, MLX-LM, text only.
2. The hard bugs (which token is the readout token, chat-template placement,
   single-token labels, position bias, calibration) are solved here and
   **frozen as fixtures** (`fixtures/`): state, questions, expected token ids,
   expected raw probabilities at a fixed model and seed.
3. The Swift + vision port is a **separate repo** whose definition of done is
   reproducing those fixtures. It swaps the model runner; readout, masking and
   calibration logic are ported unchanged.
4. The runner sits behind one small interface (`Runner`: prefill a token
   sequence against an optional cached prefix, return hidden state / logits at
   readout positions) so vision changes only the prefill path.

## Consequences
- Vision and Swift are explicit non-goals here.
- Every readout decision must be expressible as a fixture, which forces
  determinism (no sampling anywhere).
- The Swift repo owes this one nothing but fixture parity.

## Alternatives rejected
- **One repo, both languages**: tooling mismatch, and the Python side would be
  slowed by the Swift build. Rejected.
- **Swift first**: see Context. Rejected.
