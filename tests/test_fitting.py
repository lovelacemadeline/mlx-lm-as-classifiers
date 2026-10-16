"""Story 3.2 — calibrate CLI: fit T per (model, readout) from labelled decisions."""

from __future__ import annotations

import json
import math
import random

import pytest

from mlx_classifiers import cli
from mlx_classifiers.calibrate import softmax
from mlx_classifiers.decide import decide
from mlx_classifiers.fitting import CalibrationConfig, calibrate_records, label_index
from mlx_classifiers.prompt import question_options
from mlx_classifiers.runner import FakeRunner
from mlx_classifiers.schema import SystemOneRequest

TRUE_T = 3.0  # the synthetic labels are drawn as if the model were this over-confident
QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None},
}


def synthetic_records(n: int, seed: int = 0, model: str = "fake") -> list[dict]:
    """``n`` labelled decisions from the FakeRunner, labels sampled at ``TRUE_T``."""
    runner, rng = FakeRunner(model_id=model), random.Random(seed)
    out = []
    for i in range(n):
        req = SystemOneRequest.model_validate(
            {"prefix": "Route tickets.", "state": f"Ticket {i}.", "questions": {"q": QUESTION}}
        )
        ans = decide(req, runner).answers["q"]
        options = list(question_options(req.questions["q"]))
        raw = [ans.raw_probabilities[o] for o in options]
        truth = softmax([math.log(p) for p in raw], TRUE_T)
        label = rng.choices(options, weights=truth)[0]
        out.append(
            {
                "model": model,
                "readout": "first-token",
                "options": options,
                "raw_probabilities": raw,
                "label": label,
            }
        )
    return out


@pytest.fixture(scope="module")
def records():
    return synthetic_records(600)


def test_fit_recovers_the_overconfidence_and_lowers_heldout_ece(records):
    (fit,) = calibrate_records(records, heldout=0.3, seed=1)
    assert (fit.model, fit.readout, fit.n_train, fit.n_heldout) == ("fake", "first-token", 420, 180)
    assert 2.0 < fit.temperature < 4.5
    assert fit.ece_after < fit.ece_before
    assert fit.nll_after < fit.nll_before


def test_unlabelled_lines_are_skipped_and_groups_split(records):
    other = synthetic_records(40, seed=2, model="other")
    unlabelled = [{**r, "label": None} for r in records[:50]]
    fits = calibrate_records(records[:100] + other + unlabelled)
    assert [(f.model, f.n_train + f.n_heldout) for f in fits] == [("fake", 100), ("other", 40)]


def test_label_forms():
    noul = {"options": ["yes", "no"], "label": True}
    score = {"options": ["0", "1", "2"], "label": 2}
    assert label_index(noul) == 0 and label_index({**noul, "label": "no"}) == 1
    assert label_index(score) == 2
    with pytest.raises(ValueError, match="not one of"):
        label_index({"options": ["a", "b"], "label": "c"})


def test_calibrate_cli_prints_ece_and_merges_config(tmp_path, records, capsys):
    data = tmp_path / "decisions.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "calibration.json"
    CalibrationConfig({"keep-me": {"sequence": {"temperature": 0.5}}}).save(out)

    assert cli.main(["calibrate", str(data), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "ECE before" in printed and "held-out" in printed and f"wrote {out}" in printed

    cfg = CalibrationConfig.load(out)
    assert 2.0 < cfg.temperature("fake", "first-token") < 4.5
    assert cfg.temperature("keep-me", "sequence") == 0.5  # other entries survive a refit
    assert cfg.temperature("fake", "sequence") is None
    entry = cfg.models["fake"]["first-token"]
    assert entry["ece_after"] < entry["ece_before"] and "fitted_at" in entry


def test_calibrate_cli_with_nothing_labelled_fails(tmp_path, capsys):
    data = tmp_path / "d.jsonl"
    data.write_text(
        json.dumps({"model": "m", "options": ["a", "b"], "raw_probabilities": [0.5, 0.5]})
    )
    assert cli.main(["calibrate", str(data), "--out", str(tmp_path / "c.json")]) == 1
    assert not (tmp_path / "c.json").exists()


def test_server_applies_the_fitted_temperature():
    pytest.importorskip("litestar")
    from litestar.testing import TestClient

    from mlx_classifiers.server import create_app

    cfg = CalibrationConfig({"fake": {"first-token": {"temperature": 2.5}}})
    body = {
        "prefix": "Route tickets.",
        "state": "Ticket 1.",
        "questions": {"q": QUESTION},
        "options": {"debug": True},
    }
    with TestClient(app=create_app(FakeRunner(), calibration=cfg)) as client:
        a = client.post("/v1/systemone", json=body).json()["answers"]["q"]
        body["options"]["calibrated"] = False
        b = client.post("/v1/systemone", json=body).json()["answers"]["q"]
    assert a["debug"]["temperature"] == 2.5
    assert max(a["probabilities"].values()) < max(a["raw_probabilities"].values())
    assert b["debug"]["temperature"] == 1.0
    assert b["probabilities"] == b["raw_probabilities"]
