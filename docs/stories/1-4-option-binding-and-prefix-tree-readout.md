---
type: story
status: ready-for-dev
title: Option binding and prefix-tree first-token readout
owner: maddie
updated: 2026-10-16
anchor: none  # not yet built; anchors added when src/mlx_classifiers/readout.py lands
---

# Story 1.4 — Option binding and prefix-tree first-token readout

Bind each option to a label and resolve the label to token ids with the real
tokenizer at compile time. Default labels are letters A..Z; `noul` uses a fixed
yes/no label pair. Build a prefix tree over the label token sequences and read
out at the first distinguishing depth (ADR-0003). A label that is not a single
token at depth one is a hard `LabelTokenizationError`, never a silent fallback.

## Acceptance Criteria
- **Given** labels that each tokenise to one token **when** bound **then** the
  readout is depth one and the mask contains exactly one id per option.
- **Given** two labels sharing a first token **when** bound **then** the
  prefix tree reports the distinguishing depth and `readout` raises
  `LabelTokenizationError` naming both labels (depth > 1 is unsupported in
  first-token mode; the error points at sequence mode).
- **Given** a label with a leading-space variant in the tokenizer **when**
  bound **then** the variant matching the template position is chosen and the
  choice is recorded in `debug.option_token_ids`.
- **Given** more than 26 options **when** bound **then** validation fails with
  a message pointing at independent scoring (future story).
