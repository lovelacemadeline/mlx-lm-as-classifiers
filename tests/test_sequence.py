"""Story 1.6 — sequence-scoring readout mode (ADR-0003)."""

from __future__ import annotations

import math

import pytest

from mlx_classifiers import FakeRunner, SystemOneRequest, SystemOneResponse, decide
from mlx_classifiers.calibrate import token_logprob
from mlx_classifiers.errors import LabelTokenizationError
from mlx_classifiers.prompt import compile_request
from mlx_classifiers.readout import bind_labels
from mlx_classifiers.sequence import score_sequences


def _req(criteria, readout="sequence", **extra):
    return SystemOneRequest.model_validate(
        {
            "prefix": "You route tickets.",
            "state": "My invoice was charged twice.",
            "questions": {
                "dept": {"type": "choice", "instructions": "Which team?", "criteria": criteria}
            },
            "options": {"readout": readout, **extra},
        }
    )


COLLIDING = {"billing": "Payments", "billable hours": "Time tracking"}


def test_colliding_multi_token_options_get_probabilities():
    runner = FakeRunner()
    req = _req(COLLIDING)
    prompt = compile_request(req, runner)[0]
    with pytest.raises(LabelTokenizationError, match="share a first token"):
        bind_labels(prompt.labels, prompt.text, runner.tokenize)  # first-token cannot read these
    resp = decide(req, runner)
    ans = resp.answers["dept"]
    assert set(ans.probabilities) == {"billing", "billable hours"}
    assert math.fsum(ans.probabilities.values()) == pytest.approx(1.0)
    assert ans.choice in COLLIDING
    assert resp.usage.output_tokens == 0


def test_sequence_scores_are_length_normalised_teacher_forced_loglik():
    runner = FakeRunner()
    prompt = compile_request(_req(COLLIDING), runner)[0]
    scores = score_sequences(prompt, runner)
    ids = scores.token_ids[1]
    assert len(ids) > 1 and scores.token_ids[0][0] == ids[0]  # shared first token
    # Recompute option 2's loglikelihood by hand from uncached prefills.
    full = prompt.tokens + ids
    n = len(prompt.tokens)
    ll = 0.0
    for j, t in enumerate(ids):
        row = runner.prefill(full[: n + j]).at(n + j - 1)
        ll += token_logprob(row, t)
    assert scores.loglikelihoods[1] == pytest.approx(ll)
    assert scores.scores[1] == pytest.approx(ll / len(ids))


def test_sequence_prompt_lists_option_texts():
    prompt = compile_request(_req(COLLIDING), FakeRunner())[0]
    assert "- billing: Payments\n- billable hours: Time tracking" in prompt.text
    assert "exactly as written" in prompt.text
    assert prompt.labels == ("billing", "billable hours")


def test_both_modes_agree_on_single_token_labels():
    """With context-free logits and option names that equal the letter labels, both
    modes score the same token at the same position, so they must agree exactly."""
    criteria = {"A": None, "B": None, "C": None}

    def runner():
        r = FakeRunner(logit_fn=lambda digest, t: 0.0)
        (b,) = r.tokenize("B")
        r.logit_bias[b] = 3.0
        return r

    first = decide(_req(criteria, readout="first-token"), runner()).answers["dept"]
    seq = decide(_req(criteria), runner()).answers["dept"]
    assert first.choice == seq.choice == "B"
    for k in criteria:
        assert seq.probabilities[k] == pytest.approx(first.probabilities[k], abs=1e-9)


def test_both_modes_return_the_same_argmax_on_a_peaked_fake():
    criteria = {"A": None, "B": None, "C": None}
    r1, r2 = FakeRunner(), FakeRunner()
    for r in (r1, r2):
        (c,) = r.tokenize("C")
        r.logit_bias[c] = 12.0  # dominates the pseudo-random base logits
    first = decide(_req(criteria, readout="first-token"), r1).answers["dept"]
    seq = decide(_req(criteria), r2).answers["dept"]
    assert first.choice == seq.choice == "C"


def test_sequence_mode_score_noul_debug_and_wire_validity():
    req = SystemOneRequest.model_validate(
        {
            "state": "s",
            "questions": {
                "sev": {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]},
                "yn": {"type": "noul", "instructions": "Repro?"},
            },
            "options": {"readout": "sequence", "debug": True},
        }
    )
    resp = decide(req, FakeRunner(), temperature=1.5)
    sev = resp.answers["sev"]
    assert set(sev.probabilities) == {"0", "1"} and sev.legend == {"0": "low", "1": "high"}
    assert sev.debug.temperature == 1.5 and set(sev.debug.option_token_ids) == {"0", "1"}
    assert 0 <= resp.answers["yn"].noul <= 1
    SystemOneResponse.model_validate(resp.to_wire())
