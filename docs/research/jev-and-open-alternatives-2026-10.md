---
type: research
status: active
title: Jev (TypeSafe) and the open alternatives — what the mechanism actually is
owner: maddie
updated: 2026-10-16
---

# Jev and open alternatives (October 2026)

Time-boxed read-up, an input to the PRD, not authoritative.

## What TypeSafe says
Jev never emits text. It takes a `state` and named `questions`, scores
pre-declared options for all questions in one query in parallel, and returns
calibrated probabilities. Training uses a calibration-focused RL method they
call RLCD. High-cardinality choices are scored independently then chosen.
No architecture, weights or paper published. Pricing is input-only; output is
free. Latency claims are 70 to 500 ms. ([blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev))

## What independent sources say
- Speedups land around 5x to 25x versus the 193x headline
  ([DataSci Ocean](https://datasciocean.com/en/ai-concept/jev-overview/)).
- "No hallucinations" means no output outside the schema, not no wrong answers
  ([innfactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/)).
- The founder's stated moat is training data, not architecture.

## Open clones
[open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) is the
most careful: options labelled A, B, C as single tokens, several questions
packed in one sequence with placeholder answers, one forward pass, logits read
only at readout positions, softmax over the letter logits, fitted temperature
for calibration, optional order permutation. It matched Jev on a third-party
typed-decisions benchmark with better calibration, on stock Qwen3.6-27B.
Caveats it reports: packing hurts small models, 6 to 9% of packed answers
change with neighbours, and calibration must be fitted by the user. Others:
Laya (421M open weights), Nimble (LoRA on Qwen3.5-9B), Rizzo Flow (llama.cpp).

## Wire shape
Captured from gateway docs ([LiteLLM](https://docs.litellm.ai/docs/pass_through/typesafe.md),
[Zuplo](https://zuplo.com/docs/ai-gateway/jev.md)); now ADR-0001 and
[wire-schema](../reference/wire-schema.md).

## Takeaways for this repo
1. The mechanism is first-token readout at the last prefill position.
2. Calibration and position bias are the actual engineering, not the readout.
3. Prefer separate requests sharing a cached prefix over packing, on an engine
   with a prefix cache.
