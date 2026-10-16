---
type: reference
status: active
title: Wire schema — request and response for /v1/systemone
owner: maddie
updated: 2026-10-16
---

# Wire schema

The machine-readable truth is `schema/systemone-request.schema.json` and
`schema/systemone-response.schema.json` (JSON Schema 2020-12). This page
explains the shape and the definitions the schema cannot express. Fields marked
**x-local** are our extensions beyond Jev's native API (ADR-0001); a client
that ignores them is still Jev-compatible.

## Request

```json
{
  "model": "gemma-4-e2b-int8",
  "prefix": "You are the triage router for ...",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "has_repro":  { "type": "noul",   "instructions": "Does the report include repro steps?" },
    "department": { "type": "choice", "instructions": "Which team should handle this?",
                    "criteria": { "billing": "Payments, refunds", "technical": "Bugs, outages", "sales": null } },
    "severity":   { "type": "score",  "instructions": "How severe is the impact?",
                    "criteria": ["trivial", "minor", "major", "critical"] }
  },
  "options": { "permutations": 1, "calibrated": true, "readout": "first-token", "debug": false }
}
```

- `state` may be a string, a JSON object (rendered as pretty JSON), or a list
  of strings (rendered as numbered blocks). A list may also hold image blocks
  (x-local, below).
- `prefix` (x-local) is the stable part. The runner caches its KV once and
  only prefills `state` + questions per request.
- `criteria` is a **map** for `choice` (name to optional description) and an
  **ordered list** for `score` (lowest first; the index is the score).

## Response

```json
{
  "model": "gemma-4-e2b-int8",
  "answers": {
    "has_repro":  { "type": "noul",  "noul": 0.12, "raw_noul": 0.04 },
    "department": { "type": "choice", "choice": "technical",
                    "probabilities": { "billing": 0.08, "technical": 0.85, "sales": 0.07 },
                    "raw_probabilities": { "billing": 0.02, "technical": 0.96, "sales": 0.02 },
                    "confidence": 0.85 },
    "severity":   { "type": "score", "score": 2, "expected_score": 2.1,
                    "legend": { "0": "trivial", "1": "minor", "2": "major", "3": "critical" },
                    "probabilities": { "0": 0.02, "1": 0.18, "2": 0.55, "3": 0.25 },
                    "confidence": 0.55 }
  },
  "usage": { "input_tokens": 312, "output_tokens": 0, "cached_tokens": 180 },
  "timing_ms": { "prefill": 31.2, "readout": 0.4, "total": 33.0 }
}
```

## Definitions

- **probabilities** are the calibrated softmax over the declared options
  only. They sum to 1. Nothing outside the declared set can appear.
- **raw_probabilities** (x-local) are the same before temperature scaling.
  Always present so calibration is auditable.
- **confidence** is the calibrated probability of the returned `choice` /
  `score`. Jev documents a separate confidence number; we define ours
  explicitly as the top calibrated probability so there is nothing to guess.
- **noul** is the calibrated P(yes). A `noul` is a two-option choice with
  fixed labels; it exists so clients get a scalar.
- **expected_score** (x-local) is the probability-weighted mean index, for
  callers that want a continuous value from an ordinal scale.
- **output_tokens** is always 0. Nothing is decoded.
- **debug.permutation_spread** (x-local) is, per option, the max minus min
  calibrated probability across the `options.permutations` orderings of a
  `choice` question; present only when more than one ordering ran. `noul` and
  `score` questions are never permuted (fixed Yes/No pair; ordered scale).

## Readout modes (ADR-0003)

- `first-token` (default): each option is bound to a single-token label; the
  distribution at the readout position is masked to those tokens.
- `sequence`: each full option string is scored by teacher-forced
  loglikelihood from the shared prefix cache, length-normalised. Slower, for
  options whose first token does not discriminate.

## Label style (x-local, story [1.8](../stories/1-8-option-text-labels-wire-option.md))

`options.label_style` chooses what the model is asked to emit in
`first-token` mode:

- `letters` (default): options are listed as `A. name: description`, the
  answer line asks for a letter, and the letters are the bound labels.
- `option-text`: options are listed by their own name (`- name: description`,
  the same rendering as `sequence` mode), the answer line asks for one option
  exactly as written, and each `choice` name / `score` level is bound as its
  own first-token label. Every name must be a single token in context, else
  the request fails with a `LabelTokenizationError` pointing at
  `readout: "sequence"` or `label_style: "letters"`. `noul` always uses
  Yes/No.

```json
{ "options": { "label_style": "option-text" } }
```

Use it when letter labels carry a prior of their own: on small models a fixed
letter can win regardless of the state (label bias), and permutations only
flatten that towards uniform. `sequence` mode already answers by option text,
so the field has no effect there.

## Delta cache (x-local, story [3.4](../stories/3-4-delta-prefix-caching.md))

`options.delta_cache` (default `true`) lets a server reuse the KV of the
longest common token prefix with earlier prompts, so only the changed tail is
prefilled; `usage.cached_tokens` then counts those tokens too. `false` runs
the request without it (the declared `prefix` is still cached). See
[serving](serving.md#delta-prefix-caching).

```json
{ "options": { "delta_cache": false } }
```

## Image blocks (x-local, story [6.1](../stories/6-1-image-blocks-in-state.md))

A list `state` may hold `{"type": "image/png", "base64": "<PNG file, base64>"}`
blocks beside strings; no other keys. The model sees the image exactly where
the block sits: it renders as `[i] ` followed by the model's image placeholder
run (Gemma 4: `<|image>`, N soft tokens, `<image|>`, with N from the image's
size, 270 for a 160 x 192 frame, 256 for any square image).

```json
{ "state": ["Status: hp 6 | tick 3", {"type": "image/png", "base64": "iVBORw0KGgo..."}] }
```

A runner without a vision path, a payload that is not a PNG, or a state text
containing the placeholder token itself is a compile error. Requests without
image blocks compile exactly as before.

### Soft-token budget (x-local, story [6.7](../stories/6-7-vision-soft-token-budget.md))

`options.vision_soft_tokens` (70, 140 or 280; omit for the runner's budget,
280 unless the server runs `serve --vision-tokens N`) sets how many soft tokens
an image may become. Gemma 4 resizes the image, aspect kept, to at most
budget x 9 patches of 16 px (sides multiples of 48) and pools them 3 x 3, so N
is at most the budget. A 160 x 192 frame: 270 tokens from 2430 patches at 280,
120 from 1080 at 140, 63 from 567 at 70. The vision tower runs over the
patches, so its cost falls faster than N (attention is quadratic in the patch
count), and the prompt is shorter by the difference in N. The same PNG at two
budgets is two different images to the delta cache.

```json
{ "options": { "vision_soft_tokens": 140 } }
```

## Versioning

The schema `$id` carries `/v1/`. Additive x-local fields do not bump the
version. A breaking change to a Jev-native field does.
