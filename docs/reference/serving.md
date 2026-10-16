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
  question's path).
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
