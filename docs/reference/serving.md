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
- `POST /v1/chat/completions` — a thin, non-streaming OpenAI-style chat
  completion on the same weights (`messages`, `max_tokens`, `temperature`;
  `usage.prompt_tokens_details.cached_tokens`). It exists to prove the
  single-process story, not to replace a full chat server.

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
