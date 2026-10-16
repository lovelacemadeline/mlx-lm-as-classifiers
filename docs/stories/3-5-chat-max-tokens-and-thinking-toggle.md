---
type: story
status: completed
title: Chat route honours max_tokens and a per-call thinking toggle
owner: maddie
updated: 2026-10-16
anchor:
  - route: "POST /v1/chat/completions"
  - symbol: "src/mlx_classifiers/server.py::split_reasoning"
  - symbol: "src/mlx_classifiers/server.py::PrefixLRU.peek"
  - symbol: "src/mlx_classifiers/template.py::ThinkingMarkers"
  - symbol: "src/mlx_classifiers/template.py::HFChatTemplate.render_messages"
  - test: "tests/test_server.py::test_chat_honours_max_tokens_with_finish_reason_and_usage"
  - test: "tests/test_server.py::test_enable_thinking_renders_a_different_prompt_for_that_call_only"
  - test: "tests/test_server.py::test_thinking_trace_is_split_into_reasoning"
  - test: "tests/test_server.py::test_long_chat_leaves_the_decision_prefix_entry_untouched"
  - test: "tests/test_server.py::test_thinking_chat_reads_a_matching_prefix_without_writing_the_lru"
  - test: "tests/test_server.py::test_real_model_thinking_plan_call"
---

# Story 3.5 — Chat route honours max_tokens and a per-call thinking toggle

The gridworld benchmark (systemone-gridworld, Epic 5) sends its "System Two"
planning calls to this server's chat route. A plan needs up to thousands of
tokens and Gemma's thinking trace, per call; decision prompts on the same
server must keep thinking off (`HFChatTemplate.render` forces
`enable_thinking=False`). A small fixed generation cap would silently
truncate plans.

## Acceptance Criteria
- **Given** a chat body with `max_tokens` (or `max_completion_tokens`) **when**
  it is at most the server ceiling (`--max-tokens-ceiling`, default 8192)
  **then** generation runs to exactly that cap, with no lower silent cap; with
  none given the default is 512; above the ceiling, or not a positive
  integer, the route answers 422 naming the field.
- **Given** a completion **then** `finish_reason` is `stop` (end-of-turn
  token) or `length` (cap reached), and `usage` carries the real
  `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens` and
  `completion_tokens_details.reasoning_tokens`.
- **Given** `enable_thinking: true` **when** the chat renders **then** that
  call's prompt is rendered in the template's thinking mode (it differs from
  the plain render) and the next call, and every decision, is not.
- **Given** generated tokens containing the thinking markers **then** the
  trace is returned as `message.reasoning` and the rest as `message.content`
  (marker handling in [serving](../reference/serving.md#thinking-and-max_tokens)).
- **Given** a decision, then a long chat completion (plain or thinking) with
  the same system text, then the decision again **then** the second decision
  is a prefix-cache hit with identical answers, and the LRU entry is the same
  object: generation runs on its own copy of the KV, and a thinking call only
  peeks the LRU (no insert, no recency bump, no hit/miss count).

## Implementation notes
- `server.py`: `_max_tokens` validates the cap against
  `ServerState.max_tokens_ceiling`; `mlx_generate` now returns the generated
  ids (without the stop token) in `ChatResult.token_ids`; `split_reasoning`
  splits them on the template's `ThinkingMarkers` (sub-sequence match on the
  markers' token ids, so multi-token markers work too).
- `template.py`: `render_messages(..., enable_thinking=False)` on both
  templates; `render` (decisions) is untouched. `HFChatTemplate.thinking`
  defaults to `GEMMA_THINKING`; the toy template mirrors Gemma with
  `<|think|>` in the system turn and `<|channel|>`/`<|channelend|>`.
- With thinking on, Gemma 4 inserts `<|think|>` into the system turn, so a
  plan call's prefix never equals a decision's and it prefills its own
  prefix; a template that leaves the system turn alone (Qwen3) shares the
  decision entry read-only.

## Observed on E2B (2026-10-16)
`test_real_model_thinking_plan_call`, gemma-4-E2B-it-qat-4bit on this Mac,
`max_tokens: 2048`, `enable_thinking: true`, a "plan how to triage a
week-long payout outage" prompt: **both** branches held: `reasoning` was
non-empty (787 reasoning tokens, about 3300 characters) **and**
`finish_reason` was `length` (2048 completion tokens; the answer after the
trace was still going). Wall time 69.9 s for the completion, about 29 tok/s.
The decision before and after matched and the second was a hit. A trivial
prompt ("17*3?") with thinking on stopped after 3 tokens with no trace: E2B
may skip thinking when the question does not need it.
