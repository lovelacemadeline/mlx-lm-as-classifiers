"""Story 3.3 — decision log: serve -> log -> add labels -> calibrate -> serve calibrated."""

from __future__ import annotations

import json
import math
import random

import pytest

pytest.importorskip("litestar")
from litestar.testing import TestClient

from mlx_classifiers import cli
from mlx_classifiers.calibrate import softmax
from mlx_classifiers.decision_log import request_hash
from mlx_classifiers.fitting import CalibrationConfig
from mlx_classifiers.runner import FakeRunner
from mlx_classifiers.schema import SystemOneRequest
from mlx_classifiers.server import create_app

QUESTIONS = {
    "has_repro": {"type": "noul", "instructions": "Does the report include repro steps?"},
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None},
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the impact?",
        "criteria": ["trivial", "minor", "major", "critical"],
    },
}


def body(i: int) -> dict:
    return {
        "model": "triage",
        "prefix": "Route tickets.",
        "state": f"Ticket {i}.",
        "questions": QUESTIONS,
    }


def test_each_served_question_appends_one_record(tmp_path):
    log = tmp_path / "logs" / "decisions.jsonl"
    with TestClient(app=create_app(FakeRunner(), log=log)) as client:
        ok = client.post("/v1/systemone", json=body(0))
        bad = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert ok.status_code == 200 and bad.status_code == 422
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["question"] for r in lines] == list(QUESTIONS)  # the 422 logged nothing
    answers = ok.json()["answers"]
    by_q = {r["question"]: r for r in lines}
    assert {r["request_hash"] for r in lines} == {
        request_hash(SystemOneRequest.model_validate(body(0)))
    }
    for r in lines:
        assert (r["v"], r["model"], r["requested_model"]) == (1, "fake", "triage")
        assert (r["readout"], r["permutations"], r["calibrated"], r["temperature"]) == (
            "first-token",
            1,
            True,
            1.0,
        )
        assert r["timing_ms"]["total"] > 0 and "label" not in r
        assert math.isclose(sum(r["raw_probabilities"]), 1.0)
    noul, choice, score = by_q["has_repro"], by_q["department"], by_q["severity"]
    assert noul["options"] == ["yes", "no"]
    assert noul["raw_probabilities"][0] == answers["has_repro"]["raw_noul"]
    assert choice["options"] == ["billing", "technical", "sales"]
    assert choice["raw_probabilities"] == [
        answers["department"]["raw_probabilities"][o] for o in choice["options"]
    ]
    assert choice["answer"] == answers["department"]["choice"]
    assert (
        score["options"] == ["0", "1", "2", "3"] and score["answer"] == answers["severity"]["score"]
    )


def test_round_trip_serve_log_label_calibrate(tmp_path, capsys):
    log = tmp_path / "decisions.jsonl"
    with TestClient(app=create_app(FakeRunner(), log=log)) as client:
        for i in range(150):
            assert client.post("/v1/systemone", json=body(i)).status_code == 200

    # A labeller adds `label` to each line; here, sampled as if the model were
    # over-confident by T = 3, the shape a refit should recover.
    rng = random.Random(0)
    labelled = tmp_path / "labelled.jsonl"
    with labelled.open("w") as fh:
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            truth = softmax([math.log(p) for p in rec["raw_probabilities"]], 3.0)
            rec["label"] = rng.choices(rec["options"], weights=truth)[0]
            fh.write(json.dumps(rec) + "\n")

    cfg_path = tmp_path / "calibration.json"
    assert cli.main(["calibrate", str(labelled), "--out", str(cfg_path)]) == 0
    assert "450 decisions, 450 labelled" in capsys.readouterr().out
    t = CalibrationConfig.load(cfg_path).temperature("fake", "first-token")
    assert t is not None and t > 1.5

    log2 = tmp_path / "after.jsonl"
    app = create_app(FakeRunner(), calibration=CalibrationConfig.load(cfg_path), log=log2)
    with TestClient(app=app) as client:
        client.post("/v1/systemone", json=body(0))
    rec = json.loads(log2.read_text().splitlines()[1])
    assert rec["temperature"] == t
    assert max(rec["probabilities"]) < max(rec["raw_probabilities"])
