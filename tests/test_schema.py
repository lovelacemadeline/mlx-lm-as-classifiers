"""Story 1.1 — wire schema models mirror schema/*.json (ADR-0001)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from mlx_classifiers.schema import SystemOneRequest, SystemOneResponse

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"

# The examples from docs/reference/wire-schema.md, verbatim.
EXAMPLE_REQUEST: dict[str, Any] = {
    "model": "gemma-4-e2b-int8",
    "prefix": "You are the triage router for ...",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "has_repro": {"type": "noul", "instructions": "Does the report include repro steps?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, refunds",
                "technical": "Bugs, outages",
                "sales": None,
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the impact?",
            "criteria": ["trivial", "minor", "major", "critical"],
        },
    },
    "options": {"permutations": 1, "calibrated": True, "readout": "first-token", "debug": False},
}

EXAMPLE_RESPONSE: dict[str, Any] = {
    "model": "gemma-4-e2b-int8",
    "answers": {
        "has_repro": {"type": "noul", "noul": 0.12, "raw_noul": 0.04},
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
            "raw_probabilities": {"billing": 0.02, "technical": 0.96, "sales": 0.02},
            "confidence": 0.85,
        },
        "severity": {
            "type": "score",
            "score": 2,
            "expected_score": 2.1,
            "legend": {"0": "trivial", "1": "minor", "2": "major", "3": "critical"},
            "probabilities": {"0": 0.02, "1": 0.18, "2": 0.55, "3": 0.25},
            "confidence": 0.55,
        },
    },
    "usage": {"input_tokens": 312, "output_tokens": 0, "cached_tokens": 180},
    "timing_ms": {"prefill": 31.2, "readout": 0.4, "total": 33.0},
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text())


def test_example_request_round_trips():
    req = SystemOneRequest.model_validate(EXAMPLE_REQUEST)
    assert req.to_wire() == EXAMPLE_REQUEST
    assert json.loads(req.model_dump_json(exclude_unset=True)) == EXAMPLE_REQUEST


def test_example_response_round_trips():
    resp = SystemOneResponse.model_validate(EXAMPLE_RESPONSE)
    assert resp.to_wire() == EXAMPLE_RESPONSE


def test_minimal_request_omits_unset_fields():
    minimal = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    req = SystemOneRequest.model_validate(minimal)
    assert req.to_wire() == minimal
    assert req.effective_options.readout == "first-token"


@pytest.mark.parametrize(
    "state",
    ["plain text", {"b": 1, "a": {"nested": True}}, ["block one", "block two"]],
)
def test_state_shapes_accepted(state):
    req = SystemOneRequest.model_validate({**EXAMPLE_REQUEST, "state": state})
    assert req.state == state


def test_choice_with_one_criterion_fails_naming_field():
    bad = {"type": "choice", "instructions": "?", "criteria": {"only": None}}
    with pytest.raises(ValidationError) as exc:
        SystemOneRequest.model_validate({"state": "s", "questions": {"q": bad}})
    locs = [err["loc"] for err in exc.value.errors()]
    assert any("criteria" in loc for loc in locs), locs


def test_score_with_one_level_fails_naming_field():
    bad = {"type": "score", "instructions": "?", "criteria": ["only"]}
    with pytest.raises(ValidationError) as exc:
        SystemOneRequest.model_validate({"state": "s", "questions": {"q": bad}})
    locs = [err["loc"] for err in exc.value.errors()]
    assert any("criteria" in loc for loc in locs), locs


def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError) as exc:
        SystemOneRequest.model_validate({**EXAMPLE_REQUEST, "temperature": 0.7})
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


def test_unknown_question_and_option_fields_rejected():
    q = {"type": "noul", "instructions": "?", "labels": ["y", "n"]}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "s", "questions": {"q": q}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({**EXAMPLE_REQUEST, "options": {"sampling": True}})


def test_unknown_question_type_and_empty_questions_rejected():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {"state": "s", "questions": {"q": {"type": "rank", "instructions": "?"}}}
        )
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "s", "questions": {}})


def test_explicit_null_for_optional_fields_rejected():
    for key in ("model", "prefix", "options"):
        with pytest.raises(ValidationError):
            SystemOneRequest.model_validate({**EXAMPLE_REQUEST, key: None})


def test_options_bounds():
    for perms in (0, 9):
        with pytest.raises(ValidationError):
            SystemOneRequest.model_validate({**EXAMPLE_REQUEST, "options": {"permutations": perms}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({**EXAMPLE_REQUEST, "options": {"readout": "decode"}})


# --- drift tripwire: pydantic model_json_schema() vs schema/*.json ----------------------------


class _Walker:
    """Structurally compare a hand-written schema node with a pydantic-generated one.

    Checked: ``required`` sets, ``properties`` key sets, ``additionalProperties: false``,
    ``const``/``enum`` values, ``type``, and the numeric/size bounds. Pydantic's
    ``anyOf [X, null]`` for optional fields is unwrapped; ``oneOf``/``anyOf`` are treated
    alike and compared branch by branch in order.
    """

    BOUNDS = ("minimum", "maximum", "minProperties", "minItems")

    def __init__(self, ours: dict[str, Any], theirs: dict[str, Any]):
        self.ours_root, self.theirs_root = ours, theirs
        self.problems: list[str] = []

    def _resolve(self, root: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
        while "$ref" in node:
            name = node["$ref"].removeprefix("#/$defs/")
            node = root["$defs"][name]
        branches = node.get("anyOf")
        if branches is not None:
            non_null = [b for b in branches if b != {"type": "null"}]
            if len(non_null) == 1 and len(non_null) < len(branches):
                return self._resolve(root, non_null[0])
        return node

    def compare(self, file_node: dict[str, Any], py_node: dict[str, Any], path: str) -> None:
        f = self._resolve(self.ours_root, file_node)
        p = self._resolve(self.theirs_root, py_node)

        f_branches = f.get("oneOf") or f.get("anyOf")
        if f_branches is not None:
            p_branches = p.get("oneOf") or p.get("anyOf")
            if p_branches is None or len(p_branches) != len(f_branches):
                self.problems.append(f"{path}: union shape differs")
                return
            for i, (fb, pb) in enumerate(zip(f_branches, p_branches, strict=True)):
                self.compare(fb, pb, f"{path}|{i}")
            return

        for key in ("const", "enum", "type"):
            if key in f and f[key] != p.get(key):
                # ["string", "null"] in the file is pydantic's anyOf string|null.
                if key == "type" and isinstance(f[key], list):
                    continue
                self.problems.append(f"{path}: {key} {f[key]!r} != {p.get(key)!r}")
        for key in self.BOUNDS:
            if f.get(key) != p.get(key):
                self.problems.append(f"{path}: {key} {f.get(key)!r} != {p.get(key)!r}")

        if set(f.get("required", [])) != set(p.get("required", [])):
            self.problems.append(
                f"{path}: required {sorted(f.get('required', []))} != "
                f"{sorted(p.get('required', []))}"
            )
        f_props, p_props = f.get("properties", {}), p.get("properties", {})
        if set(f_props) != set(p_props):
            self.problems.append(f"{path}: properties {sorted(f_props)} != {sorted(p_props)}")
        for name in sorted(set(f_props) & set(p_props)):
            self.compare(f_props[name], p_props[name], f"{path}.{name}")

        f_add, p_add = f.get("additionalProperties"), p.get("additionalProperties")
        if f_add is False and p_add is not False:
            self.problems.append(f"{path}: additionalProperties false not mirrored")
        if f_add is not False and p_add is False:
            self.problems.append(f"{path}: pydantic forbids extras the file allows")
        if isinstance(f_add, dict):
            if not isinstance(p_add, dict):
                self.problems.append(f"{path}: additionalProperties schema missing")
            else:
                self.compare(f_add, p_add, f"{path}{{*}}")
        if "items" in f:
            self.compare(f["items"], p.get("items", {}), f"{path}[]")


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("systemone-request.schema.json", SystemOneRequest),
        ("systemone-response.schema.json", SystemOneResponse),
    ],
)
def test_pydantic_schema_matches_json_schema_files(filename, model):
    file_schema = _load(filename)
    walker = _Walker(file_schema, model.model_json_schema())
    walker.compare(file_schema, model.model_json_schema(), file_schema["title"])
    assert walker.problems == []


def test_drift_walker_catches_a_missing_required_field():
    """The tripwire itself trips: drop a required field from the file copy."""
    file_schema = _load("systemone-response.schema.json")
    file_schema["$defs"]["ChoiceAnswer"]["required"].append("rationale")
    walker = _Walker(file_schema, SystemOneResponse.model_json_schema())
    walker.compare(file_schema, SystemOneResponse.model_json_schema(), "resp")
    assert any("required" in p for p in walker.problems)


def test_every_def_name_exists_in_pydantic():
    for filename, model in (
        ("systemone-request.schema.json", SystemOneRequest),
        ("systemone-response.schema.json", SystemOneResponse),
    ):
        ours = set(_load(filename)["$defs"]) - {"Question", "Answer"}  # unions, not classes
        theirs = set(model.model_json_schema()["$defs"])
        assert ours <= theirs, ours - theirs


def test_answer_union_discriminates():
    adapter = TypeAdapter(dict[str, Any])
    data = adapter.validate_python(EXAMPLE_RESPONSE)
    resp = SystemOneResponse.model_validate(data)
    assert type(resp.answers["severity"]).__name__ == "ScoreAnswer"
    assert type(resp.answers["has_repro"]).__name__ == "NoulAnswer"
