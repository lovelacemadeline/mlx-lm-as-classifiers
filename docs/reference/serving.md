---
type: reference
status: active
title: Serving — one process for /v1/systemone and chat completions
owner: maddie
updated: 2026-10-16
---

# Serving

`mlx-classifiers serve` loads one model and serves two routes from it
(story [3.1](../stories/3-1-systemone-route.md)). Install the extra first:
`uv pip install -e '.[server]'` (litestar + uvicorn).

```bash
mlx-classifiers serve --model mlx-community/gemma-4-E2B-it-qat-4bit --port 8080
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "prefix": "You are a triage router.", "state": "The app crashes on login.",
  "questions": {"dept": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": null, "technical": null}}}}'
```

## Routes

- `POST /v1/systemone` — the wire shape in [wire-schema](wire-schema.md).
  A body that does not validate as a `SystemOneRequest` (or is not JSON), or a
  request the prompt compiler rejects (e.g. a label that is not one token),
  answers **422** with `{"detail": ..., "errors": [{"loc", "msg", "type"}]}`,
  where `loc` is the dotted field path (`options.permutations`,
  `questions.department.choice.criteria`; the union member name is part of a
  question's path). `options` pass through to `decide` unchanged, including
  `options.label_style` (story [1.8](../stories/1-8-option-text-labels-wire-option.md));
  an option-text name that is not one token answers 422 the same way. The
  calibration temperature is keyed by `(model, readout)` only, so an
  option-text request uses the first-token temperature.
- `POST /v1/chat/completions` — a non-streaming OpenAI-style chat
  completion on the same weights (`messages`, `max_tokens`, `temperature`,
  `enable_thinking`). Not a full chat server (no streaming, tools or
  sampling knobs beyond temperature); see
  [Thinking and max_tokens](#thinking-and-max_tokens) for what it honours.

## Thinking and max_tokens

Story [3.5](../stories/3-5-chat-max-tokens-and-thinking-toggle.md). A
planning call (the gridworld's System Two) sends:

```json
{"messages": [{"role": "system", "content": "<the decision prefix text>"},
              {"role": "user", "content": "<the planning prompt>"}],
 "max_tokens": 4096, "enable_thinking": true, "temperature": 0}
```

- **`max_tokens`** (or `max_completion_tokens`, which wins if both are sent):
  default 512. Honoured exactly up to the server ceiling
  `serve --max-tokens-ceiling` (default 8192); above it, or not a positive
  integer, answers **422** naming the field and the ceiling. Never capped
  silently.
- **`finish_reason`**: `stop` when the model emits an end-of-turn id (the
  tokenizer's `eos_token_ids`; Gemma 4: `<eos>`, `<turn|>`, and id 50),
  `length` when `max_tokens` tokens were generated.
- **`usage`**: `prompt_tokens` (whole rendered prompt), `completion_tokens`
  (generated, including the stop token), `total_tokens`,
  `prompt_tokens_details.cached_tokens` (prefix tokens reused) and
  `completion_tokens_details.reasoning_tokens`.
- **`enable_thinking`** (bool, default false; anything else is 422): renders
  this call's prompt with the chat template's `enable_thinking=True`. Gemma 4
  puts `<|think|>\n` at the top of the system turn. Decisions always render
  with thinking off, whatever chats ask for.

**Marker handling.** The route splits the generated token ids (not the
text) on the template's markers; Gemma 4's are `<|channel>` (id 100) and
`<channel|>` (id 101), with the channel name `thought\n` after the start:

1. No `<|channel>` in the output: `message.content` is the whole text, there
   is no `message.reasoning` key, `reasoning_tokens` is 0.
2. `<|channel>` ... `<channel|>`: `message.reasoning` is the text between
   them with a leading `thought\n` dropped; `message.content` is the text
   outside the span (normally only what follows `<channel|>`). Both are
   whitespace-trimmed. `reasoning_tokens` counts the span including both
   markers.
3. `<|channel>` with no `<channel|>` (cut off by `max_tokens`, so
   `finish_reason` is `length`): everything after the start marker is
   `reasoning`, `content` is `""`.

The split applies whenever the markers appear, thinking requested or not.

**KV.** Generation runs on a deep copy of the prefix KV, never the LRU's
entry. A plain chat uses the LRU like a decision (it may insert its system
turn). A thinking chat only *peeks*: it uses an entry already keyed by its
rendered system-turn tokens, but never inserts one, bumps recency or counts a
hit or miss, so a planning call cannot evict a decision prefix. With Gemma 4
the thinking system turn differs from the decision's (`<|think|>`), so a plan
call prefills its own prefix (`x-prefix-cache: miss`); a template that leaves
the system turn alone (Qwen3) shares the decision's entry read-only.

## The shared prefix cache

Both routes take their system-turn KV from one LRU (`PrefixLRU`, default 16
entries, `--prefix-cache-size`) keyed by the prefix **token ids**. A decision's
`prefix` and a chat's leading `system` message render to the same system turn
(`ChatTemplate.render_messages` matches `render`), so a chat completion and a
decision with the same system text share one prefilled prefix, in either
order. Entries are deep-copied before being extended, never mutated.

Every response carries `x-prefix-cache: hit | miss | none` (none: no system
turn). `usage.cached_tokens` counts the prefix tokens each prefill reused, so
it is non-zero on a first request too; the header is what says whether this
request paid for the prefix.

## Delta prefix caching

Story [3.4](../stories/3-4-delta-prefix-caching.md). Beside the declared
`prefix` LRU, the runner keeps a delta cache (`PromptCacheStore`, default 8
entries and 2 GiB of KV; `serve --delta-cache-entries N --delta-cache-mb M`,
`--no-delta-cache` turns it off). Every prefill looks up the stored token
sequence sharing the **longest common token prefix** with its own prompt,
rewinds a copy of that KV to the shared length, and prefills only the rest;
afterwards it stores its own state. A sequential controller whose state
changes only near the end (a map with a moving tail, an append-only
transcript) then pays per tick for the delta, not the whole state. Other
questions and permutations of the same request share the state head, so they
benefit automatically.

- `usage.cached_tokens` counts the tokens each prefill took from cache
  (declared prefix or delta hit, whichever is longer).
- `x-prefix-cache-tokens: N` is the hit length of the request's first
  prefill. `x-prefix-cache: hit | miss | none` is unchanged and still
  reports only the declared prefix's LRU (clients compare it to `hit`).
- `options.delta_cache: false` runs one request without the delta cache (no
  lookup, no insert); its declared prefix is still cached.
- Sliding-window layers (Gemma's 512-token `RotatingKVCache`) can only be
  rewound as far as the rows they still hold: an entry rewinds into its own
  last delta, or anywhere if it was built from an empty cache in one chunk.
  A deeper rewind is refused for that entry and the lookup tries the next
  one, then falls back to the declared prefix. Changing the state earlier
  than the previous tick did is the case that hits this.
- Results match the uncached path up to bf16 noise from the different
  chunking (story 3.4 measures it); golden fixtures run with the delta cache
  off because they pin a cache path.
- On a MoE model (Gemma 4 26B-A4B) a ~100-token delta still touches every
  expert, and that sets the per-decision floor. `MLXRunner.load` swaps the
  experts to a per-expert kernel that is bit-identical to the stock path
  ([3.6](../stories/3-6-delta-cache-fixed-overhead.md); `moe_kernel=False`
  keeps the stock one).

All model work runs on one dedicated worker thread (MLX arrays are bound to the
stream of the thread that created them, so the model is loaded there too). That
also serialises requests: one model, one GPU.

## Calibration config

`mlx-classifiers calibrate <decisions.jsonl>...` (story
[3.2](../stories/3-2-calibrate-cli.md)) fits one temperature per
`(model, readout)` and writes it to a JSON config. `serve --calibration PATH`
loads it; both default to `$MLX_CLASSIFIERS_CALIBRATION`, else
`./calibration.json` (a missing file means no temperatures: calibrated equals
raw). Refits merge: entries for other models or modes are kept.

```json
{"version": 1,
 "models": {"mlx-community/gemma-4-E2B-it-qat-4bit":
   {"first-token": {"temperature": 1.84, "n_train": 700, "n_heldout": 300,
                    "ece_before": 0.091, "ece_after": 0.023,
                    "nll_before": 0.61, "nll_after": 0.48,
                    "fitted_at": "2026-10-16T12:00:00+00:00"}}}}
```

Only `temperature` is read back. The model key is the server's loaded model id
(`MLXRunner.model_id`), not the request's `model` string. A request with
`options.calibrated: false` is served at `T = 1`.

**Input** is JSONL, one answered question per line (the decision log
format, below) with a `label` added. The fields read are `model`, `readout`,
`options` (option names in declared order: criteria keys, `"0"`.. for score
levels, `"yes"`/`"no"` for noul), `raw_probabilities` (aligned with
`options`) and `label` (an option name; a bool for noul, an int index for
score). Lines without a label are skipped. `log(raw_probabilities)` is the
option-logit row up to a constant, so fitting on it is fitting on logits. Each
group is split with a seeded shuffle (`--heldout 0.3 --seed 0`); the printed
ECE (`--bins 15`) and NLL before and after are on the held-out split.

## Decision log

`serve --log PATH` appends one JSONL line per answered question of every
served decision (story [3.3](../stories/3-3-decision-log.md)); requests that
answer 422 log nothing. Add a `label` to a line and it is calibrate input.

```json
{"v": 1, "ts": "2026-10-16T12:00:00.123+00:00", "request_hash": "sha256:...",
 "model": "mlx-community/gemma-4-E2B-it-qat-4bit", "requested_model": "gemma-4-e2b-int8",
 "readout": "first-token", "permutations": 1, "calibrated": true, "temperature": 1.84,
 "question": "department", "type": "choice", "options": ["billing", "technical", "sales"],
 "raw_probabilities": [0.02, 0.96, 0.02], "probabilities": [0.08, 0.85, 0.07],
 "answer": "technical", "timing_ms": {"prefill": 31.2, "readout": 0.4, "total": 33.0}}
```

- `request_hash` is `sha256:` of the request's canonical JSON (sorted keys,
  no whitespace); the lines of one request share it, and its `timing_ms`.
- `model` is the served model id (the calibration key); `requested_model`
  is the request's `model` string, present only when given.
- `temperature` is the one applied (1 when uncalibrated or none fitted);
  `probabilities` are what was served, `raw_probabilities` the `T = 1` row.
- `answer` is the served choice, score index, or for noul `"yes"` when the
  calibrated P(yes) is at least 0.5.

Refit loop: serve with `--log`, label lines, `mlx-classifiers calibrate
labelled.jsonl`, restart serve (it reads the config at startup).
