"""Story 1.5 — masked softmax, temperature calibration, confidence; and ``decide``."""

from __future__ import annotations

import math
import random

import pytest

from mlx_classifiers import FakeRunner, SystemOneRequest, SystemOneResponse, decide
from mlx_classifiers.answers import build_answer
from mlx_classifiers.calibrate import (
    TemperatureScaler,
    expected_calibration_error,
    gather,
    nll,
    softmax,
)
from mlx_classifiers.prompt import compile_request
from mlx_classifiers.readout import bind_options

REQUEST = {
    "prefix": "You are the triage router.",
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
}


def _req(**options) -> SystemOneRequest:
    body = dict(REQUEST)
    if options:
        body["options"] = options
    return SystemOneRequest.model_validate(body)


def _bound(name: str, runner: FakeRunner | None = None):
    runner = runner or FakeRunner()
    prompt = next(p for p in compile_request(_req(), runner) if p.name == name)
    binding = bind_options(prompt, runner.tokenize)
    row = runner.prefill(prompt.tokens).at(prompt.readout_position)
    return prompt, binding, row


def test_softmax_is_masked_to_declared_options():
    prompt, binding, row = _bound("department")
    answer = build_answer(prompt, binding, row)
    assert set(answer.probabilities) == {"billing", "technical", "sales"}
    assert math.fsum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    # Exactly the softmax over the three gathered logits, nothing from the rest of the vocab.
    expected = softmax(gather(row, binding.token_ids))
    assert list(answer.probabilities.values()) == expected


def test_temperature_changes_calibrated_not_raw():
    prompt, binding, row = _bound("department")
    plain = build_answer(prompt, binding, row)
    warm = build_answer(prompt, binding, row, temperature=2.5)
    assert warm.raw_probabilities == plain.raw_probabilities == plain.probabilities
    assert warm.probabilities != plain.probabilities
    assert warm.choice == plain.choice  # temperature is monotone
    assert warm.confidence == warm.probabilities[warm.choice] == max(warm.probabilities.values())


def test_noul_is_calibrated_p_yes():
    prompt, binding, row = _bound("has_repro")
    ans = build_answer(prompt, binding, row, temperature=2.0)
    logits = gather(row, binding.token_ids)
    assert ans.noul == softmax(logits, 2.0)[0]
    assert ans.raw_noul == softmax(logits)[0]


def test_score_is_argmax_and_expected_score_is_weighted_mean():
    prompt, binding, row = _bound("severity")
    ans = build_answer(prompt, binding, row)
    probs = [ans.probabilities[str(i)] for i in range(4)]
    assert ans.score == probs.index(max(probs))
    assert ans.expected_score == pytest.approx(sum(i * p for i, p in enumerate(probs)))
    assert ans.legend == {"0": "trivial", "1": "minor", "2": "major", "3": "critical"}
    assert ans.confidence == probs[ans.score]


def test_biased_runner_picks_the_biased_option():
    runner = FakeRunner()
    (b,) = runner.tokenize("B")
    runner.logit_bias[b] = 20.0
    _, binding, _ = _bound("department", runner)
    resp = decide(_req(), runner)
    assert resp.answers["department"].choice == "technical"  # option B
    assert resp.answers["department"].confidence > 0.99
    assert binding.option_token_ids["technical"] == [b]


def _overconfident(n=3000, k=4, scale=3.0, seed=0):
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        z = [rng.gauss(0.0, 1.5) for _ in range(k)]
        p = softmax(z)
        labels.append(rng.choices(range(k), weights=p)[0])
        rows.append([scale * x for x in z])  # the "model" is 3x too sharp
    return rows, labels


def test_temperature_scaler_fixes_overconfidence():
    rows, labels = _overconfident()
    scaler = TemperatureScaler()
    report = scaler.fit(rows, labels)
    assert report.temperature > 1.0
    assert report.temperature == pytest.approx(3.0, rel=0.15)
    assert report.ece_after < report.ece_before
    assert report.nll_after < report.nll_before
    assert scaler.apply(rows[0]) == softmax(rows[0], report.temperature)
    assert TemperatureScaler().fit(rows, labels) == report  # deterministic


def test_temperature_scaler_minimises_nll():
    rows, labels = _overconfident(n=500, seed=1)
    t = TemperatureScaler().fit(rows, labels).temperature
    best = nll(rows, labels, t)
    assert best <= nll(rows, labels, t * 1.05) and best <= nll(rows, labels, t / 1.05)


def test_ece_of_perfectly_calibrated_bins_is_zero():
    rows = [[0.75, 0.25]] * 4
    assert expected_calibration_error(rows, [0, 0, 0, 1]) == pytest.approx(0.0)
    assert expected_calibration_error([[1.0, 0.0]], [1]) == pytest.approx(1.0)


# --- decide() end to end on the FakeRunner -------------------------------------------------


def test_decide_one_pass_per_question_no_decode():
    runner = FakeRunner()
    resp = decide(_req(), runner)
    assert isinstance(resp, SystemOneResponse)
    assert set(resp.answers) == {"has_repro", "department", "severity"}
    assert resp.usage.output_tokens == 0
    assert runner.projected_rows == 3  # one readout row per question, nothing else
    prompts = compile_request(_req(), runner)
    prefix_len = prompts[0].prefix_len
    assert resp.usage.cached_tokens == 3 * prefix_len
    assert resp.usage.input_tokens == sum(len(p.tokens) - prefix_len for p in prompts)
    SystemOneResponse.model_validate(resp.to_wire())  # wire-valid


def test_decide_is_deterministic():
    a = decide(_req(debug=True), FakeRunner()).to_wire()
    b = decide(_req(debug=True), FakeRunner()).to_wire()
    a.pop("timing_ms"), b.pop("timing_ms")
    assert a == b


def test_decide_calibrated_flag_and_debug():
    t = 2.0
    cal = decide(_req(debug=True), FakeRunner(), temperature=t)
    raw = decide(_req(calibrated=False, debug=True), FakeRunner(), temperature=t)
    dc, dr = cal.answers["department"], raw.answers["department"]
    assert dc.raw_probabilities == dr.raw_probabilities == dr.probabilities
    assert dc.probabilities != dc.raw_probabilities
    assert dc.debug.temperature == t and dr.debug.temperature == 1.0
    assert set(dc.debug.option_token_ids) == {"billing", "technical", "sales"}
    assert set(dc.debug.raw_logits) == {"billing", "technical", "sales"}
    assert dc.debug.readout_position > 0
    assert decide(_req(), FakeRunner()).answers["department"].debug is None
