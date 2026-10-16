"""Epic 5 — the static decision battery: format and seed set (5.1), runner (5.2)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from mlx_classifiers.battery import (
    BatteryError,
    load_battery,
    parse_case,
    render_markdown,
    run_battery,
    run_case,
    summarise,
)
from mlx_classifiers.cli import main as cli_main
from mlx_classifiers.fitting import CalibrationConfig
from mlx_classifiers.prompt import question_options
from mlx_classifiers.runner import FakeRunner
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


# --- 5.2: the battery runner ---------------------------------------------------


def _a_runner():
    """Always prefers label "A": picks whichever option is displayed first."""
    runner = FakeRunner(logit_fn=lambda digest, t: 0.0)
    (a,) = runner.tokenize("A")
    runner.logit_bias[a] = 4.0
    return runner


def _choice(case_id, gold, prefix="P", **tags):
    return _case(
        id=case_id,
        prefix=prefix,
        questions={
            "q": {
                "type": "choice",
                "instructions": "i",
                "criteria": {"a": None, "b": None, "c": None},
            }
        },
        gold={"q": gold},
        tags={"source": "t", "difficulty": "easy", **tags},
    )


def _write(tmp_path, objs):
    p = tmp_path / "b.jsonl"
    p.write_text("".join(json.dumps(o) + "\n" for o in objs))
    return p


def test_battery_on_fake_runner_reports_every_metric_and_round_trips():
    cases = load_battery(SEED)
    report = run_battery(cases, FakeRunner(), permutations=2)
    s = report.summary
    assert report.problems == []
    assert s["n_questions"] == len(cases)
    assert set(s["by_type"]) == {"choice", "noul", "score", "all"}
    for row in s["by_type"].values():
        assert 0 <= row["accuracy"] <= 1 and 0 <= row["ece_raw"] <= 1
        assert row["ece_calibrated"] is None  # no calibration config given
    assert {"source", "difficulty", "variant"} <= set(s["by_tag"])
    assert s["by_tag"]["source"]["agent-robot"]["n"] == 59
    assert s["battery_fit"] and s["battery_fit"][0]["readout"] == "first-token"
    assert s["position_bias"]["n"] == 223 and s["position_bias"]["orderings"] == 2
    assert s["perturbation"]["groups"] == 30 and s["perturbation"]["variants"] == 79
    md = render_markdown(report)
    assert "| choice | 223 |" in md and "## Position bias" in md and "## Prefix perturbation" in md


def test_accuracy_ece_and_position_bias_on_a_first_slot_runner(tmp_path):
    """A runner that always says "A" is right exactly when gold is declared first,
    and every rotation moves its answer: shift near 1, flip rate 1."""
    p = _write(tmp_path, [_choice("x1", "a"), _choice("x2", "a"), _choice("x3", "b")])
    report = run_battery(load_battery(p), _a_runner(), permutations=3)
    s = report.summary
    assert s["by_type"]["choice"]["accuracy"] == pytest.approx(2 / 3)
    conf = s["by_type"]["choice"]["mean_confidence"]
    assert s["by_type"]["choice"]["ece_raw"] == pytest.approx(abs(2 / 3 - conf))
    assert s["position_bias"]["flip_rate"] == 1.0
    assert s["position_bias"]["mean_shift"] > 0.9
    r = report.results[0]
    assert r.order_flips == 2 and r.orderings == 2


def test_perturbation_flip_rate_counts_variants_that_change_answer(tmp_path):
    def member(case_id, variant, prefix):
        return _choice(case_id, "a", prefix=prefix, group="g", variant=variant)

    objs = [member("g", "base", "P0"), member("g~p1", "p1", "P1"), member("g~p2", "p2", "P2")]
    p = _write(tmp_path, objs)
    steady = run_battery(load_battery(p), _a_runner()).summary["perturbation"]
    assert steady["variant_flip_rate"] == 0.0 and steady["flipped"] == []

    results, _ = run_case(load_battery(p)[0], _a_runner())
    base_pred = results[0].pred
    # Fake the variants' predictions to differ from the base and summarise.
    rs = [results[0]] + [
        replace(results[0], case_id=c, tags={**results[0].tags, "variant": v}, pred=base_pred + 1)
        for c, v in (("g~p1", "p1"), ("g~p2", "p2"))
    ]
    pt = summarise(rs, "fake")["perturbation"]
    assert pt["variant_flip_rate"] == 1.0 and pt["group_flip_rate"] == 1.0
    assert pt["flipped"][0]["gold"] == "a"


def test_calibration_config_gives_calibrated_ece(tmp_path):
    cfg = tmp_path / "cal.json"
    CalibrationConfig({"fake": {"first-token": {"temperature": 3.0}}}).save(cfg)
    out = tmp_path / "runs"
    rc = cli_main(
        [
            "battery",
            str(SEED),
            "--runner",
            "fake",
            "--calibration",
            str(cfg),
            "--permutations",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    (md,) = out.glob("battery-*.md")
    (js,) = out.glob("battery-*.json")
    body = json.loads(js.read_text())
    assert body["env"]["model"] == "fake" and "git_rev" in body["env"]
    assert "lowpowermode" in body["env"]
    assert body["config"]["temperatures"] == {"first-token": 3.0}
    row = body["summary"]["by_type"]["all"]
    assert row["ece_calibrated"] is not None and row["ece_calibrated"] != row["ece_raw"]
    assert len(body["results"]) == 343 and body["problems"] == []
    assert "git:" in md.read_text()
