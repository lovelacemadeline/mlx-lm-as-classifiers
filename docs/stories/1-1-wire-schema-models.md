---
type: story
status: completed
title: Wire schema models and validation
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/schema.py::SystemOneRequest"
  - symbol: "src/mlx_classifiers/schema.py::SystemOneResponse"
  - symbol: "src/mlx_classifiers/schema.py::ChoiceQuestion"
  - test: "tests/test_schema.py::test_example_request_round_trips"
  - test: "tests/test_schema.py::test_choice_with_one_criterion_fails_naming_field"
  - test: "tests/test_schema.py::test_pydantic_schema_matches_json_schema_files"
  - test: "tests/test_schema.py::test_unknown_top_level_field_rejected"
---

# Story 1.1 — Wire schema models and validation

Pydantic v2 models mirroring `schema/systemone-request.schema.json` and
`schema/systemone-response.schema.json`: `NoulQuestion`, `ChoiceQuestion`,
`ScoreQuestion`, the `Request`, and the three `Answer` types. The JSON Schema
files are the contract; the models must round-trip them.

## Acceptance Criteria
- **Given** the example request in [wire-schema](../reference/wire-schema.md)
  **when** parsed **then** it validates and re-serialises to equal JSON.
- **Given** a `choice` with fewer than two criteria, or a `score` with fewer
  than two levels **when** parsed **then** validation fails with the field named.
- **Given** the pydantic models **when** exported with `model_json_schema()`
  **then** a test asserts every required field and enum in `schema/*.json` is
  present (drift tripwire between the two).
- **Given** an unknown top-level field **when** parsed **then** it is rejected
  (`additionalProperties: false` mirrored).

## Implementation notes
- Model class names equal the `$defs` names in `schema/*.json`; the inline
  `timing_ms` object is `Timing`, `usage` is `Usage`.
- The drift tripwire walks both schemas structurally (required sets, property
  sets, `additionalProperties: false`, const/enum/type, numeric and size
  bounds), unwrapping pydantic's `anyOf [X, null]`. A companion test mutates
  the file copy to prove the tripwire trips.
- Optional request fields (`model`, `prefix`, `options`) can be omitted but not
  sent as `null`, matching the file schema. `to_wire()` on a request drops
  unset fields (exact round-trip); on a response it drops `None` fields.
