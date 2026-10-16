---
type: story
status: completed
title: Option binding and prefix-tree first-token readout
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/readout.py::bind_options"
  - symbol: "src/mlx_classifiers/readout.py::PrefixTree"
  - symbol: "src/mlx_classifiers/errors.py::LabelTokenizationError"
  - test: "tests/test_readout.py::test_single_token_labels_bind_at_depth_one"
  - test: "tests/test_readout.py::test_shared_first_token_reports_depth_and_raises_naming_both"
  - test: "tests/test_readout.py::test_leading_space_variant_matches_template_position"
  - test: "tests/test_readout.py::test_gemma_picks_bare_variant_after_turn_marker"
  - test: "tests/test_readout.py::test_more_than_26_options_fails_pointing_at_independent_scoring"
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

## Implementation notes
- **In-context resolution.** A label's ids are `tokenize(prompt + emitted)`
  minus `tokenize(prompt)`, where `emitted` is the bare label after whitespace
  (every real template's marker ends in a newline) and `" " + label`
  otherwise. If appending the label re-tokenizes the prompt's tail, that is a
  `LabelTokenizationError` too: no next-token distribution at the readout
  position can represent it. On Gemma 4 the bare `A` (not `" A"`) is chosen
  after `<|turn>model\n`; the chosen ids are exposed as
  `OptionBinding.option_token_ids` and copied into `debug.option_token_ids`
  by `decide` (story 1.5).
- **Errors, in order:** duplicate labels; labels sharing a first token (names
  every colliding group and the distinguishing depth, points at
  `options.readout = "sequence"`); then any label longer than one token.
  `PrefixTree.depth` is `None` only for identically tokenized labels.
- **More than 26 options** raises `TooManyOptionsError` (an
  `OptionBindingError`) at compile time, from `option_labels`, rather than in
  pydantic validation: the wire schema has no option cap, so adding one to the
  models would break the story 1.1 drift test.
- Letters A..Z and Yes/No are verified single tokens at depth one on both the
  Gemma 4 and Qwen3 tokenizers.
