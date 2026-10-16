---
type: story
status: completed
title: /v1/systemone route beside chat completions
owner: maddie
updated: 2026-10-16
anchor:
  - route: "POST /v1/systemone"
  - route: "POST /v1/chat/completions"
  - symbol: "src/mlx_classifiers/server.py::create_app"
  - symbol: "src/mlx_classifiers/server.py::PrefixLRU"
  - symbol: "src/mlx_classifiers/server.py::CachingRunner"
  - symbol: "src/mlx_classifiers/template.py::HFChatTemplate.render_messages"
  - test: "tests/test_server.py::test_systemone_returns_a_schema_valid_response"
  - test: "tests/test_server.py::test_invalid_request_is_422_with_field_paths"
  - test: "tests/test_server.py::test_second_request_with_same_prefix_hits_the_lru"
  - test: "tests/test_server.py::test_chat_and_decision_share_one_prefix_cache"
  - test: "tests/test_server.py::test_real_model_decision_and_chat_share_prefix"
---

# Story 3.1 — /v1/systemone route beside chat completions

A `POST /v1/systemone` route on the same server process as chat completions,
sharing the loaded model and the prefix cache. Returns the wire schema
response; validation errors return 422 with field paths.

## Acceptance Criteria
- **Given** a valid request **when** posted **then** the response validates
  against `schema/systemone-response.schema.json` (via the pinned
  `SystemOneResponse` mirror) and `output_tokens` is 0.
- **Given** an invalid request **when** posted **then** the answer is 422 and
  each error names its dotted field path.
- **Given** two decisions with the same `prefix` **when** both run **then** the
  second is an LRU hit (`x-prefix-cache: hit`) and reports `cached_tokens` > 0.
- **Given** a chat completion and a decision with the same system text
  **when** both run **then** the second reuses the first's prefix KV, and
  neither mutates it (the same decision answers the same afterwards).

## Implementation notes
- `src/mlx_classifiers/server.py`: a litestar app (`create_app`) over one
  runner; `PrefixLRU` keyed by prefix token ids; `CachingRunner` wraps the
  runner so `decide` runs unchanged and its `cache_prefix` hits the LRU. The
  chat route renders with the new `ChatTemplate.render_messages` (same system
  turn as `render`) and generates with `mlx_lm.stream_generate` from a
  deep copy of the cached prefix (`mlx_generate`).
- litestar over `mlx_lm.server`: mlx_lm's server is a stdlib `http.server`
  with its own prompt cache that `decide` cannot reach; wrapping it would mean
  patching its handler. The litestar app is one small module and the routes
  resolve as docloom anchors.
- MLX binds arrays to the creating thread's stream, so all model work (and the
  load, via `create_app(load=...)`) runs on one worker thread.
- `mlx-classifiers serve [--model --host --port --prefix-cache-size]`. Route
  and cache behaviour are documented in
  [serving](../reference/serving.md).
