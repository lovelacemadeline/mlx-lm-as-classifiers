---
type: story
status: completed
title: Per-fixture gate — near-tied gridworld cases on answer and ids, clear-winner image fixtures at every budget
owner: maddie
updated: 2026-10-17
anchor:
  - symbol: "src/mlx_classifiers/fixtures.py::GATES"
  - symbol: "src/mlx_classifiers/fixtures.py::FixtureCase"
  - symbol: "src/mlx_classifiers/fixtures.py::build_fixture"
  - symbol: "src/mlx_classifiers/fixtures.py::compare"
  - symbol: "src/mlx_classifiers/fixtures.py::check"
  - symbol: "src/mlx_classifiers/fixtures.py::CheckReport"
  - symbol: "src/mlx_classifiers/fixtures.py::SEED_CASES"
  - test: "tests/test_fixtures.py::test_gate_key_only_on_answer_and_ids_fixtures"
  - test: "tests/test_fixtures.py::test_clear_winner_image_fixtures_cover_every_budget_with_a_wide_margin"
  - test: "tests/test_fixtures.py::test_answer_and_ids_reports_dp_without_gating_it"
  - test: "tests/test_fixtures.py::test_answer_and_ids_still_gates_ids_images_and_answer"
  - test: "tests/test_fixtures.py::test_unknown_gate_in_a_fixture_fails_the_case"
  - test: "tests/test_fixtures.py::test_committed_fixtures_reproduce_on_mlx"
---

# Story 4.3 — Per-fixture gate, and clear-winner image fixtures

The Swift port (mlx-swift-as-classifiers story 4.1) reproduces every image
fixture except the three gridworld action cases
(`choice-first-token-image-gridworld{,-vt140,-vt70}`). Those have six
near-tied action logits (story [6.5](6-5-image-fixtures-and-battery.md): top-1
margin 0.17 in probability), so one bf16 rounding step moves probabilities by
several percent, and MLX core 0.32.0 (which wrote the fixtures) and 0.32.2
(mlx-swift) round differently. Cross-check: this repo's own Python on MLX
0.32.2 fails `image-gridworld` (max |dp| 0.0757) and `-vt140` (0.021) at the
1e-3 gate. That is a property of the cases, not of either port.

Decision (Maddie): keep the gridworld cases as answer-and-ids checks, and
replace their probability coverage with image fixtures that have a clear
winner at all three budgets.

## The gate field

A fixture may carry a top-level `gate`: `"full"` (the default, and the key is
then absent, so every existing full-gate file is byte-identical) or
`"answer-and-ids"`. Under `answer-and-ids` the checker keeps every exact
comparison (tokens, `prefix_len`, readout position, options, labels, option
token ids, `images`, `answer`, `error.type`) and reports max |dp| and
max |dlogit| without gating either, even with `--logit-tol`. An unknown gate
value fails the case. The key sits beside `readout` / `prefix_cached` rather
than inside `expected`, because `expected` is the result shape a port writes
and a port does not produce a gate. `FIXTURE_FORMAT` stays
`mlx-classifiers-fixture/1`: the field is optional and an older checker that
ignores it is only stricter. Documented in
[fixture-format](../reference/fixture-format.md).

## The new fixtures

First attempt, as asked: frames rendered by the gridworld renderer
(`gridworld.render.image.to_png`, tile 16) where the move is unambiguous: the
goal adjacent N/S/E/W with the other three sides walled or open, a walled
corridor, the goal two cells east, a monster adjacent (oracle `attack`), and a
5 x 3 map at tiles 16 and 32. Fourteen frames, the six-action question and the
gridworld's colour-key state. E2B has no policy from the image: the answer is
`north` or `south` on every frame whatever the right move, with a top-1 logit
gap of at most 1.12 at 280 and at most 0.75 at 140 and 70 (often 0). A
six-action gridworld question cannot meet a 1.5 margin on E2B.

So the replacements keep the gridworld frame and change the question to one
with a clear answer: "What colour is the square that represents you?"
(red / green / blue / yellow) on the same frame `gw-s3-t00`, same prefix and
colour-key state, first-token, at budgets 280 / 140 / 70. Reusing the frame
keeps the 160 x 192 geometry and its 270 / 120 / 63 soft-token runs, so the
budget path stays covered under the full gate. The answer is `blue`, which the
colour key in the state also says ("you are the blue square"), so this case
pins numerics through the image path rather than vision accuracy; the colour
fixture is the one whose answer only the image decides.

**How much margin is enough.** A first version asked "Is there a monster (a
red cell) on the map?" (yes by 4.25 to 4.6 logits, p(yes) about 0.99). It
passed on golden at MLX 0.32.0 and 0.32.2 with max |dp| 0, but failed the
Swift port at 70 (1.6e-3, logit deltas 0 and -0.125) and story 2.3's INT8 gate
at 70 (6.3e-3 against 5e-3). At p near 0.99, one bf16 step (0.125) on one
logit moves p by p(1-p) x 0.125, about 1.7e-3. A 1e-3 gate needs a gap of
about 6 or more, like the colour fixture's 6.5. Questions tried on gridworld
frames, with their smallest gap over the three budgets: the six-action
question (0 to 0.75), goal direction (1.4, wrong answer), red-cell count
(0.12), "is this a photo of a cat" (3.5), "is there a monster" (3.75), "what
kind of picture" (4.25), the walls' colour (5.6), the goal's colour (6.1), the
player's colour (6.1, chosen).

Raw logits from the committed fixtures (golden, MLX 0.32.0):

| fixture | budget (soft tokens) | blue | runner-up | gap | p(blue) |
|---|---|---|---|---|---|
| `choice-first-token-image-gridworld-you` | 280 (270) | 26.375 | yellow 19.875 | 6.5 | 0.99676 |
| `choice-first-token-image-gridworld-you-vt140` | 140 (120) | 26.25 | yellow 20.125 | 6.125 | 0.99664 |
| `choice-first-token-image-gridworld-you-vt70` | 70 (63) | 26.375 | yellow 20.25 | 6.125 | 0.99552 |

## Results

- `mlx-classifiers fixtures generate fixtures` wrote 16 fixtures; the only
  diffs to existing files are the three `"gate": "answer-and-ids"` lines.
- `mlx-classifiers fixtures check fixtures --runner mlx`, golden venv (MLX
  0.32.0): 16/16 pass, every distance 0.
- Same check in a scratch venv with MLX 0.32.2: 16/16 pass. The new cases:
  max |dp| 0 (280), 4.5e-4 (140), 0 (70). The gridworld action cases report
  0.0757 (280), 0.021 (140), 0 (70) and pass on answer and ids; the other
  cases are within 2.6e-4.
- The Swift port (mlx-swift-as-classifiers, MLX core 0.32.2) on these files:
  the new cases at 1.8e-4 (280), 4.5e-4 (140), 6.6e-4 (70), all within 1e-3.
- `pytest` (GPU tests included): 281 passed, 6 skipped. The INT8 fixture gate
  of story 2.3 passes on the new cases.

## Acceptance Criteria
- **Given** a fixture with `gate: "answer-and-ids"` **when** a result has the right ids, images and answer but probabilities off by 0.05 **then** the case passes and reports the distance; the same result fails under the full gate. *(test_answer_and_ids_reports_dp_without_gating_it)*
- **Given** an answer-and-ids fixture **when** a token id, an image hash or the answer differs **then** the case fails. *(test_answer_and_ids_still_gates_ids_images_and_answer)*
- **Given** the committed fixtures **then** only the three gridworld action cases carry a gate, and the three clear-winner cases cover budgets 280/140/70 with a top-1 logit gap of at least 6 (1.5 was the starting target; 4 proved too small, above). *(test_gate_key_only_on_answer_and_ids_fixtures, test_clear_winner_image_fixtures_cover_every_budget_with_a_wide_margin)*
- **Given** an unknown gate value **then** the case fails naming it. *(test_unknown_gate_in_a_fixture_fails_the_case)*
- **Given** `--runner mlx` on MLX 0.32.0 and 0.32.2 **then** all 16 fixtures pass. *(test_committed_fixtures_reproduce_on_mlx; the 0.32.2 run is recorded above)*
