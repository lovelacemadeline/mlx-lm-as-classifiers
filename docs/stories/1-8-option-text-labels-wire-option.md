---
type: story
status: completed
title: Option-text labels as a wire option
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/schema.py::RequestOptions"
  - symbol: "src/mlx_classifiers/prompt.py::names_options"
  - symbol: "src/mlx_classifiers/prompt.py::option_labels"
  - symbol: "src/mlx_classifiers/readout.py::bind_options"
  - symbol: "src/mlx_classifiers/fixtures.py::run_case"
  - test: "tests/test_label_style.py::test_label_style_defaults_to_letters_and_is_closed"
  - test: "tests/test_label_style.py::test_option_text_renders_names_and_asks_for_the_name"
  - test: "tests/test_label_style.py::test_option_text_binds_the_names_as_first_token_labels"
  - test: "tests/test_label_style.py::test_option_text_labels_follow_a_permuted_order"
  - test: "tests/test_label_style.py::test_multi_token_name_raises_pointing_at_sequence_or_letters"
  - test: "tests/test_label_style.py::test_noul_keeps_yes_no"
  - test: "tests/test_label_style.py::test_score_uses_single_token_level_names_else_errors"
  - test: "tests/test_server.py::test_label_style_passes_through_to_decide"
  - test: "tests/test_mlx_integration.py::test_option_text_labels_bind_to_the_bare_name_tokens"
  - test: "tests/test_fixtures.py::test_multi_token_label_case_records_error_and_label_ids"
---

# Story 1.8 — Option-text labels as a wire option

Add the x-local request option `options.label_style: "letters" | "option-text"`
(default `letters`). With `option-text`, first-token mode shows each `choice`
option / `score` level by its own name instead of a letter, asks for the name,
and binds the names themselves as first-token labels (story 1.4's binding,
unchanged). `noul` keeps Yes/No.

**Why.** On a gridworld task Gemma 4 E2B with letter labels A–F answered
"attack" at 0.6 on every tick and never moved: a prior on the letter, not a
reading of the state. Permutations (story 1.7) only flatten that bias towards
uniform. Binding the action names as the labels made it play sensibly; all six
action words are single Gemma tokens in bare, leading-space and capitalised
forms. The fixtures module (story 4.1) already did this privately, by
compiling with the sequence-mode rendering and then binding first-token; this
story makes it a wire option and moves the fixtures onto it.

## Acceptance Criteria
- **Given** a request without `options.label_style` **when** validated **then**
  it is `letters`; any value other than `letters` / `option-text` is rejected,
  and the JSON Schema and pydantic model agree (story 1.1 drift test).
- **Given** `label_style: "option-text"` and a `choice` **when** compiled
  **then** options are bulleted by name (`- name: description`, no letters), the
  answer line is `Answer with one option exactly as written.`, and the labels
  are the option names in display order (they follow a permuted order).
- **Given** option names that are single tokens in context **when** decided
  **then** each name's in-context token id is the mask id, reported in
  `debug.option_token_ids`, and the probabilities are the masked softmax over
  those ids; on Gemma 4 E2B `up`/`down`/`left`/`right` bind to their bare
  tokens `[1048]`, `[3843]`, `[989]`, `[898]` after `<|turn>model\n`.
- **Given** a name that is not one token **when** decided **then**
  `LabelTokenizationError` is raised, naming the label and pointing at
  `options.readout = "sequence"` or `options.label_style = "letters"`; the
  server answers 422.
- **Given** a `noul` question **when** compiled with option-text **then** it
  still uses Yes/No. **Given** a `score` **then** its level names are the
  labels if single-token, else the same error.
- **Given** the server **when** a request carries `options.label_style` **then**
  it reaches `decide` unchanged.

## Implementation notes
- **Rendering is shared with sequence mode.** `prompt.names_options(readout,
  label_style)` is true in sequence mode or with option-text, and selects both
  the option-text labels (`option_labels`) and the bulleted rendering
  (`render_user_turn`). `CompiledPrompt.label_style` records the choice;
  `bind_options` uses it to pick the error hint (`OPTION_TEXT_HINT`).
- **No 26-option cap** under option-text: the cap exists because of the
  alphabet (story 1.4); names are only limited by being distinct single tokens.
- **Calibration** stays keyed by `(model, readout)` (story 3.2): an option-text
  request uses the first-token temperature. Fitting a separate one per label
  style is not in scope.
- **Fixtures.** `run_case` now sets `options.label_style` on the case's request
  (`FixtureCase.label_style`, which must agree with the request if it sets the
  option) and compiles through the normal path. The rendered prompt for the
  option-text case is byte-identical to before, so tokens, ids and expected
  values did not change and no fixture was regenerated for them; only the
  error message string in `choice-first-token-multi-token-label-error.json`
  was refreshed to the new hint (ports are not required to match messages).
