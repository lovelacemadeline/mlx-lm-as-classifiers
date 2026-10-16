---
type: story
status: completed
title: Prompt compilation — chat template, prefix/state split, readout position
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/prompt.py::compile_question"
  - symbol: "src/mlx_classifiers/prompt.py::CompiledPrompt"
  - symbol: "src/mlx_classifiers/template.py::HFChatTemplate"
  - symbol: "src/mlx_classifiers/template.py::ToyChatTemplate"
  - test: "tests/test_prompt.py::test_prefix_tokens_are_a_stable_strict_prefix"
  - test: "tests/test_prompt.py::test_choice_user_turn_lists_labelled_options_and_reads_out_last_token"
  - test: "tests/test_prompt.py::test_real_template_readout_follows_assistant_marker"
  - test: "tests/test_prompt.py::test_json_state_is_sorted_pretty_json"
---

# Story 1.3 — Prompt compilation — chat template, prefix/state split, readout position

Turn a `Request` into token ids with a known readout position per question.
Renders `prefix` as the system turn (cacheable), `state` (string, JSON object,
or string list) and one question as the user turn, and opens the assistant
turn so that the next token is the answer label. One compiled prompt per
question by default (separate mode); packed mode is out of scope here.

## Acceptance Criteria
- **Given** a request with a `prefix` **when** compiled **then** the prefix
  token ids are a strict prefix of the full sequence, byte-stable across
  requests with different `state`.
- **Given** a `choice` question **when** compiled **then** the rendered user
  turn lists each option with its label and description, and the readout
  position is the last token of the sequence.
- **Given** a Gemma or Qwen chat template **when** compiled **then** the token
  immediately before the readout position is the template's assistant-start
  marker (test with the real tokenizers, skipped if not installed).
- **Given** `state` as a JSON object **when** rendered **then** it is
  pretty-printed with sorted keys (determinism for fixtures).

## Implementation notes
- **Readout position.** The sequence ends with the template's assistant-start
  marker (Gemma 4 `<|turn>model\n`, Qwen `<|im_start|>assistant\n`); the
  readout position is the marker's last token, whose logits predict the answer
  label. The real-template test asserts the sequence ends with the tokenized
  generation prompt and that the readout position is the last index.
- **Chat template on the Runner.** `ChatTemplate` (`template.py`) renders a
  (system, user) pair into the full text plus its cacheable prefix text. The
  Runner protocol gained a `chat_template` property so compilation needs only a
  Runner; `HFChatTemplate` adapts any HF tokenizer and forces
  `enable_thinking=False` (Gemma 4 otherwise injects `<|think|>` into the
  system turn).
- **Prefix split.** The full text is tokenized once; the prefix text is
  tokenized separately and must be a token-aligned head of it, else
  `PromptError` (the prefix could not be KV-cached). Both real templates pass
  the stronger check that natural tokenization equals the split.
- **User turn.** `State:` block, `Question:` line, then `Options:` (choice,
  `A. name: description`) or `Scale (lowest to highest):` (score) with letter
  labels, or `Answer Yes or No only.` (noul). An `order` argument permutes the
  displayed options while `CompiledPrompt.label_for` keeps labels bound to
  options (used by story 1.7).
- Real-tokenizer tests load only the tokenizer from the local HF cache
  (`tests/conftest.py`) and skip when it is absent.
