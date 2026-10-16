---
type: story
status: ready-for-dev
title: Prompt compilation — chat template, prefix/state split, readout position
owner: maddie
updated: 2026-10-16
anchor: none  # not yet built; anchors added when src/mlx_classifiers/prompt.py lands
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
