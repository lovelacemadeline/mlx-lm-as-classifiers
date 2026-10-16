"""Story 2.2 — MLXRunner on the real Gemma 4 E2B (skipped when the HF cache lacks it)."""

from __future__ import annotations

import pytest
from conftest import GEMMA, HF_HUB

mx = pytest.importorskip("mlx.core")

from mlx_classifiers import SystemOneRequest, decide
from mlx_classifiers.mlx_runner import MLXRunner

_SNAP = HF_HUB / f"models--{GEMMA.replace('/', '--')}" / "snapshots"
pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        not any(_SNAP.glob("*/model.safetensors")), reason=f"{GEMMA} weights not in HF cache"
    ),
]


@pytest.fixture(scope="module")
def runner():
    return MLXRunner.load(GEMMA)


QUESTIONS = {
    "about_payments": {"type": "noul", "instructions": "Is the text about payments?"},
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "Payments, refunds, invoices, payouts",
            "technical": "Bugs, crashes, outages",
            "sales": "Pricing questions and new contracts",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the customer impact?",
        "criteria": ["trivial", "minor", "major", "critical"],
    },
}
STATE = (
    "URGENT: none of our payouts have arrived for three days and our whole business is "
    "blocked. Every payment to our merchants is failing and we are losing thousands per hour."
)
PREFIX = "You are the support triage router for a payments company. Answer precisely."


def _req(readout="first-token", state=STATE):
    return SystemOneRequest.model_validate(
        {
            "prefix": PREFIX,
            "state": state,
            "questions": QUESTIONS,
            "options": {"readout": readout, "debug": True},
        }
    )


def test_three_question_types_answer_the_obvious(runner):
    resp = decide(_req(), runner)
    a = resp.answers
    assert a["about_payments"].noul > 0.8
    assert a["department"].choice == "billing"
    assert a["severity"].score >= 2  # major or critical
    assert resp.usage.output_tokens == 0
    assert resp.usage.cached_tokens > 0
    assert resp.timing_ms.prefill > 0 and resp.timing_ms.total > 0


def test_obviously_no(runner):
    resp = decide(_req(state="The office plants need watering on Fridays."), runner)
    assert resp.answers["about_payments"].noul < 0.2


def test_raw_probabilities_identical_across_runs(runner):
    a, b = decide(_req(), runner), decide(_req(), runner)
    for name in ("department", "severity"):
        assert a.answers[name].raw_probabilities == b.answers[name].raw_probabilities
    assert a.answers["about_payments"].raw_noul == b.answers["about_payments"].raw_noul


def test_first_token_and_sequence_modes_agree_on_argmax(runner):
    ft, seq = decide(_req(), runner), decide(_req("sequence"), runner)
    assert ft.answers["department"].choice == seq.answers["department"].choice
    assert ft.answers["severity"].score == seq.answers["severity"].score
    assert (ft.answers["about_payments"].noul > 0.5) == (seq.answers["about_payments"].noul > 0.5)


def test_split_matches_full_forward(runner):
    """The split path agrees with model(all) at the readout row (bf16 noise)."""
    tokens = runner.tokenize(runner.chat_template.render(PREFIX, STATE).text)
    split = runner.prefill(tokens).at(len(tokens) - 1)
    full = runner.model(mx.array(tokens)[None])[0, -1].astype(mx.float32)
    ids = mx.argsort(full)[-20:].tolist()  # the full forward's top 20
    assert max(abs(split[i] - full[i].item()) for i in ids) < 1.0
    assert max(ids, key=split.__getitem__) == int(mx.argmax(full).item())


def test_prefix_longer_than_sliding_window(runner):
    """RotatingKVCache (window 512): a 700+ token cached prefix, deep-copied and
    extended, matches the uncached prefill and leaves the shared cache untouched."""
    long_prefix = PREFIX + " " + " ".join(f"Rule {i}: be accurate." for i in range(120))
    text = runner.chat_template.render(long_prefix, STATE).text
    prefix_text = runner.chat_template.render(long_prefix, STATE).prefix_text
    tokens, prefix = runner.tokenize(text), runner.tokenize(prefix_text)
    assert len(prefix) > 600 and tuple(tokens[: len(prefix)]) == tuple(prefix)
    cache = runner.cache_prefix(prefix)
    offsets = [c.offset for c in cache.state]
    last = len(tokens) - 1
    cached = runner.prefill(tokens, cache).at(last)
    again = runner.prefill(tokens, cache).at(last)
    uncached = runner.prefill(tokens).at(last)
    assert [c.offset for c in cache.state] == offsets
    full = runner.model(mx.array(tokens)[None])[0, -1].astype(mx.float32)
    probe = mx.argsort(full)[-20:].tolist()
    for i in probe:
        assert cached[i] == again[i]
        assert abs(cached[i] - uncached[i]) < 1.0
    top = max(probe, key=uncached.__getitem__)
    assert max(probe, key=cached.__getitem__) == top


def test_2k_prompt_projects_only_the_readout_row(runner):
    tokens = [1000 + (i * 7919) % 200_000 for i in range(2048)]
    runner.head_calls.clear()
    readout = runner.prefill(tokens)
    assert readout.positions == (2047,)
    assert len(runner.head_calls) == 1 and runner.head_calls[0][:2] == (1, 1)
