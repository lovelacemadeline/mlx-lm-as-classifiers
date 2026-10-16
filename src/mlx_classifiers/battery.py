"""The static decision battery: labelled wire requests (Epic 5).

Story 5.1 owns the format and its loader. A battery is JSONL; each line is a
wire-schema :class:`~mlx_classifiers.schema.SystemOneRequest` plus three
reserved keys, stripped before the request is validated:

- ``id`` -- unique case id;
- ``gold`` -- per question name, the right answer: an option name for
  ``choice``, ``true``/``false`` for ``noul`` (true = yes), a level index for
  ``score``;
- ``tags`` -- string labels; ``source`` and ``difficulty`` are required.
  Perturbation groups add ``group`` (shared by every member, all with the same
  gold) and ``variant`` (``base`` for the unperturbed member).

``battery/seed.jsonl`` is the seed set, built by ``battery/build_seed.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .prompt import question_options
from .schema import ChoiceQuestion, NoulQuestion, SystemOneRequest

__all__ = [
    "RESERVED_KEYS",
    "BatteryCase",
    "BatteryError",
    "gold_index",
    "load_battery",
    "parse_case",
]

RESERVED_KEYS = ("id", "gold", "tags")
REQUIRED_TAGS = ("source", "difficulty")


class BatteryError(ValueError):
    """A battery line is malformed: bad JSON, an invalid request, or a bad gold."""


@dataclass(frozen=True)
class BatteryCase:
    """One battery line. ``wire`` is the request exactly as written (reserved keys
    removed); ``gold`` maps each question name to its gold option index in
    :func:`~mlx_classifiers.prompt.question_options` order."""

    id: str
    request: SystemOneRequest
    wire: dict[str, Any]
    gold: dict[str, int]
    tags: dict[str, str]

    @property
    def group(self) -> str | None:
        return self.tags.get("group")


def gold_index(question: Any, gold: Any) -> int:
    """The index of ``gold`` among ``question``'s options; :class:`BatteryError` if
    it is not one of them or is the wrong kind for the question type."""
    options = question_options(question)
    if isinstance(question, NoulQuestion):
        if not isinstance(gold, bool):
            raise BatteryError(f"noul gold must be true or false, got {gold!r}")
        return 0 if gold else 1
    if isinstance(question, ChoiceQuestion):
        if not isinstance(gold, str) or gold not in options:
            raise BatteryError(f"choice gold {gold!r} is not one of {list(options)}")
        return options.index(gold)
    if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(options):
        raise BatteryError(f"score gold {gold!r} is not a level index 0..{len(options) - 1}")
    return gold


def parse_case(obj: Mapping[str, Any]) -> BatteryCase:
    """Validate one decoded battery line."""
    if not isinstance(obj, Mapping):
        raise BatteryError("a battery line must be a JSON object")
    case_id, gold, tags = obj.get("id"), obj.get("gold"), obj.get("tags")
    if not isinstance(case_id, str) or not case_id:
        raise BatteryError("missing or empty 'id'")
    wire = {k: v for k, v in obj.items() if k not in RESERVED_KEYS}
    try:
        request = SystemOneRequest.model_validate(wire)
    except ValidationError as exc:
        raise BatteryError(f"{case_id}: not a valid request: {exc}") from None
    if not isinstance(gold, Mapping) or set(gold) != set(request.questions):
        raise BatteryError(
            f"{case_id}: gold must name exactly the questions {sorted(request.questions)}"
        )
    if not isinstance(tags, Mapping) or any(
        not isinstance(tags.get(t), str) for t in REQUIRED_TAGS
    ):
        raise BatteryError(f"{case_id}: tags must include string {list(REQUIRED_TAGS)}")
    if any(not isinstance(v, str) for v in tags.values()):
        raise BatteryError(f"{case_id}: tag values must be strings")
    try:
        golds = {name: gold_index(q, gold[name]) for name, q in request.questions.items()}
    except BatteryError as exc:
        raise BatteryError(f"{case_id}: {exc}") from None
    return BatteryCase(case_id, request, wire, golds, dict(tags))


def load_battery(path: str | Path) -> list[BatteryCase]:
    """Every case in a battery file. Raises :class:`BatteryError` naming
    ``file:line`` on the first bad line, a duplicate id, or a perturbation group
    whose members disagree on gold."""
    cases: list[BatteryCase] = []
    seen: set[str] = set()
    group_gold: dict[str, dict[str, int]] = {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            case = parse_case(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise BatteryError(f"{path}:{n}: not JSON ({exc.msg})") from None
        except BatteryError as exc:
            raise BatteryError(f"{path}:{n}: {exc}") from None
        if case.id in seen:
            raise BatteryError(f"{path}:{n}: duplicate id {case.id!r}")
        seen.add(case.id)
        if case.group is not None:
            want = group_gold.setdefault(case.group, case.gold)
            if want != case.gold:
                raise BatteryError(f"{path}:{n}: {case.id} disagrees with group {case.group}")
        cases.append(case)
    return cases
