"""Epic 5 — the static decision battery: format and seed set (5.1)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from mlx_classifiers.battery import BatteryError, load_battery, parse_case
from mlx_classifiers.prompt import question_options
from mlx_classifiers.schema import SystemOneRequest

SEED = Path(__file__).resolve().parents[1] / "battery" / "seed.jsonl"


def _lines():
    return [json.loads(raw) for raw in SEED.read_text().splitlines() if raw.strip()]


def test_seed_battery_validates():
    """Every seed line is a valid request once the reserved keys are stripped, and
    every gold is among the options the line declares."""
    lines = _lines()
    cases = load_battery(SEED)
    assert len(cases) == len(lines) >= 300
    for obj, case in zip(lines, cases, strict=True):
        wire = {k: v for k, v in obj.items() if k not in ("id", "gold", "tags")}
        assert SystemOneRequest.model_validate(wire).to_wire() == wire, case.id
        for name, q in case.request.questions.items():
            options = question_options(q)
            gold = obj["gold"][name]
            if q.type == "choice":
                assert gold in q.criteria, case.id
            elif q.type == "noul":
                assert isinstance(gold, bool), case.id
            else:
                assert isinstance(gold, int) and 0 <= gold < len(q.criteria), case.id
            assert 0 <= case.gold[name] < len(options)


def test_seed_battery_covers_every_slice():
    cases = load_battery(SEED)
    types = Counter(q.type for c in cases for q in c.request.questions.values())
    assert types["choice"] >= 100 and types["noul"] >= 50 and types["score"] >= 50
    sources = Counter(c.tags["source"] for c in cases)
    assert sources["hand-routing"] >= 60
    assert {"arc-challenge", "hand-noul", "hand-score", "agent-robot", "agent-router"} <= set(
        sources
    )


def test_seed_perturbation_groups_share_gold_and_have_variants():
    cases = load_battery(SEED)
    groups: dict[str, list] = {}
    for c in cases:
        if c.group:
            groups.setdefault(c.group, []).append(c)
    assert len(groups) >= 25
    for gid, members in groups.items():
        variants = [m.tags["variant"] for m in members]
        assert variants.count("base") == 1 and 2 <= len(variants) - 1 <= 3, gid
        assert len({json.dumps(m.gold) for m in members}) == 1, gid
        assert len({m.request.prefix for m in members}) == len(members), gid
    near = [g for g, m in groups.items() if m[0].tags["difficulty"] == "near-miss"]
    assert len(near) >= 15


def test_seed_contains_the_probe_near_miss():
    """Story 5.1's probe: 'The robot is in a room.' + 'Battery 80%.' flipped left to
    forward. That exact pair is in the battery with gold ``left``."""
    by_id = {c.id: c for c in load_battery(SEED)}
    case = by_id["robot-05~p1"]
    assert "The robot is in a room." in case.request.prefix
    assert "Battery 80%." in case.request.state
    assert case.wire["questions"]["action"]["criteria"]["left"] and case.gold == {"action": 1}


def _case(**over):
    base = {
        "id": "x",
        "state": "s",
        "questions": {
            "q": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None}}
        },
        "gold": {"q": "a"},
        "tags": {"source": "t", "difficulty": "easy"},
    }
    return {**base, **over}


@pytest.mark.parametrize(
    "over",
    [
        {"gold": {"q": "c"}},
        {"gold": {}},
        {"gold": {"q": "a", "extra": "b"}},
        {"tags": {"source": "t"}},
        {"id": ""},
        {"questions": {}},
        {"surprise": 1},
        {
            "questions": {"q": {"type": "noul", "instructions": "i"}},
            "gold": {"q": "yes"},
        },
        {
            "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["l", "h"]}},
            "gold": {"q": 2},
        },
        {
            "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["l", "h"]}},
            "gold": {"q": True},
        },
    ],
)
def test_parse_case_rejects_bad_lines(over):
    with pytest.raises(BatteryError):
        parse_case(_case(**over))


def test_load_battery_rejects_duplicates_and_group_disagreement(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps(_case()) + "\n" + json.dumps(_case()) + "\n")
    with pytest.raises(BatteryError, match="duplicate"):
        load_battery(p)
    tags = {"source": "t", "difficulty": "easy", "group": "g"}
    a = _case(id="a", tags={**tags, "variant": "base"})
    b = _case(id="b", tags={**tags, "variant": "p1"}, gold={"q": "b"})
    p.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(BatteryError, match="disagrees"):
        load_battery(p)
