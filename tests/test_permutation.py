"""Story 1.7 — order permutation for position-bias reduction."""

from __future__ import annotations

import math

import pytest

from mlx_classifiers import FakeRunner, SystemOneRequest, SystemOneResponse, decide
from mlx_classifiers.calibrate import gather, softmax
from mlx_classifiers.decide import permutation_orders
from mlx_classifiers.prompt import compile_question
from mlx_classifiers.readout import bind_options

CRITERIA = {"billing": "Payments", "technical": "Bugs", "sales": None}


def _req(permutations=2, readout="first-token", debug=False, **extra_questions):
    return SystemOneRequest.model_validate(
        {
            "prefix": "You route tickets.",
            "state": "Charged twice for one invoice.",
            "questions": {
                "dept": {"type": "choice", "instructions": "Which team?", "criteria": CRITERIA},
                **extra_questions,
            },
            "options": {"permutations": permutations, "readout": readout, "debug": debug},
        }
    )


def _manual_distribution(order, runner, temperature=1.0):
    """Compile one ordering by hand and return its calibrated distribution."""
    prompt = compile_question(
        _req(), "dept", template=runner.chat_template, tokenize=runner.tokenize, order=order
    )
    binding = bind_options(prompt, runner.tokenize)
    row = runner.prefill(prompt.tokens).at(prompt.readout_position)
    return prompt, binding, softmax(gather(row, binding.token_ids), temperature)


def test_two_permutations_average_two_bound_distributions():
    runner = FakeRunner()
    orders = permutation_orders(3, 2, seed="dept")
    assert orders == [(0, 1, 2), (1, 2, 0)]
    (p0, b0, d0), (p1, b1, d1) = (_manual_distribution(o, runner) for o in orders)
    # Two different prompts; labels stay bound to options across orderings.
    assert p0.tokens != p1.tokens
    assert b0.labels == ("A", "B", "C") and b1.labels == ("C", "A", "B")
    assert "A. technical: Bugs" in p1.text and "C. billing: Payments" in p1.text
    ans = decide(_req(debug=True), FakeRunner()).answers["dept"]
    for i, opt in enumerate(CRITERIA):
        assert ans.probabilities[opt] == pytest.approx((d0[i] + d1[i]) / 2, abs=1e-12)
        assert ans.debug.permutation_spread[opt] == pytest.approx(abs(d0[i] - d1[i]))
    assert math.fsum(ans.probabilities.values()) == pytest.approx(1.0)
    # debug lists the label id each ordering used for each option.
    assert ans.debug.option_token_ids["billing"] == [b0.token_ids[0], b1.token_ids[0]]


def test_permutations_cost_one_prefill_each_and_share_the_prefix_cache():
    runner = FakeRunner()
    resp = decide(_req(permutations=3, yn={"type": "noul", "instructions": "Repro?"}), runner)
    assert runner.projected_rows == 3 + 1  # three orderings of dept, one noul
    assert resp.usage.output_tokens == 0 and resp.usage.cached_tokens > 0
    SystemOneResponse.model_validate(resp.to_wire())


def test_position_bias_is_averaged_out():
    """A runner that always prefers label "A" picks whichever option sits first; with
    as many rotations as options every option gets "A" once, so the mean is uniform."""
    runner = FakeRunner(logit_fn=lambda digest, t: 0.0)
    (a,) = runner.tokenize("A")
    runner.logit_bias[a] = 4.0
    biased = decide(_req(permutations=1), runner).answers["dept"]
    assert biased.choice == "billing" and biased.confidence > 0.9
    fair = decide(_req(permutations=3), runner).answers["dept"]
    for p in fair.probabilities.values():
        assert p == pytest.approx(1 / 3, abs=1e-12)


def test_permutation_orders_deterministic_and_valid():
    a = permutation_orders(4, 8, seed="q")
    assert a == permutation_orders(4, 8, seed="q")
    assert a[:4] == [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
    assert all(sorted(o) == [0, 1, 2, 3] for o in a)


def test_permutations_in_sequence_mode():
    resp = decide(_req(permutations=2, readout="sequence", debug=True), FakeRunner())
    ans = resp.answers["dept"]
    assert set(ans.probabilities) == set(CRITERIA)
    assert set(ans.debug.permutation_spread) == set(CRITERIA)


def test_single_ordering_reports_no_spread():
    ans = decide(_req(permutations=1, debug=True), FakeRunner()).answers["dept"]
    assert ans.debug.permutation_spread is None
