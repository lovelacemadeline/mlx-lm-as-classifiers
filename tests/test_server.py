"""Story 3.1 — /v1/systemone beside /v1/chat/completions, one process, one prefix LRU."""

from __future__ import annotations

import pytest
from conftest import GEMMA, HF_HUB

pytest.importorskip("litestar")
from litestar.testing import TestClient

from mlx_classifiers.runner import FakeRunner
from mlx_classifiers.schema import SystemOneResponse
from mlx_classifiers.server import ChatResult, PrefixLRU, create_app

PREFIX = "You are the support triage router for a payments company."
QUESTIONS = {
    "has_repro": {"type": "noul", "instructions": "Does the report include repro steps?"},
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages", "sales": None},
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the impact?",
        "criteria": ["trivial", "minor", "major", "critical"],
    },
}


def body(state="My payouts have failed for 3 days.", **extra):
    return {"prefix": PREFIX, "state": state, "questions": QUESTIONS, **extra}


def fake_generate(runner, tokens, prefix_cache, max_tokens, temperature):
    return ChatResult(text="ok", completion_tokens=1, finish_reason="stop")


@pytest.fixture
def app():
    return create_app(FakeRunner(), generate=fake_generate)


def lru(app) -> PrefixLRU:
    return app.state.server.prefixes


def test_systemone_returns_a_schema_valid_response(app):
    with TestClient(app=app) as client:
        r = client.post("/v1/systemone", json=body())
    assert r.status_code == 200
    resp = SystemOneResponse.model_validate(r.json())  # the pinned mirror of the schema
    assert set(resp.answers) == set(QUESTIONS)
    assert resp.usage.output_tokens == 0
    assert resp.usage.input_tokens > 0
    assert resp.model == "fake"


def test_label_style_passes_through_to_decide(app):
    """Story 1.8: ``options.label_style`` reaches ``decide`` unchanged."""
    moves = {
        "type": "choice",
        "instructions": "Which action?",
        "criteria": {"up": None, "down": None, "wait": None},
    }
    long_names = {"department": QUESTIONS["department"]}
    with TestClient(app=app) as client:
        ok = client.post(
            "/v1/systemone",
            json={
                **body(),
                "questions": {"move": moves},
                "options": {"label_style": "option-text", "debug": True},
            },
        )
        bad = client.post(
            "/v1/systemone",
            json={**body(), "questions": long_names, "options": {"label_style": "option-text"}},
        )
    assert ok.status_code == 200
    ids = ok.json()["answers"]["move"]["debug"]["option_token_ids"]
    runner = FakeRunner()
    assert all(len(v) == 1 for v in ids.values()) and ids["up"] != runner.tokenize("A")
    assert bad.status_code == 422 and 'label_style = \\"letters\\"' in bad.text


def test_invalid_request_is_422_with_field_paths(app):
    bad = body()
    bad["questions"] = {
        "department": {"type": "choice", "instructions": "x", "criteria": {"a": None}}
    }
    bad["options"] = {"permutations": 0}
    with TestClient(app=app) as client:
        r = client.post("/v1/systemone", json=bad)
        not_json = client.post(
            "/v1/systemone", content=b"{nope", headers={"content-type": "application/json"}
        )
    assert r.status_code == 422
    locs = {e["loc"] for e in r.json()["errors"]}
    assert "questions.department.choice.criteria" in locs
    assert "options.permutations" in locs
    assert not_json.status_code == 422


def test_second_request_with_same_prefix_hits_the_lru(app):
    with TestClient(app=app) as client:
        first = client.post("/v1/systemone", json=body())
        second = client.post("/v1/systemone", json=body(state="Totally different state."))
    assert first.headers["x-prefix-cache"] == "miss"
    assert second.headers["x-prefix-cache"] == "hit"
    assert second.json()["usage"]["cached_tokens"] > 0
    assert (lru(app).hits, lru(app).misses, len(lru(app))) == (1, 1, 1)


def test_chat_and_decision_share_one_prefix_cache(app):
    chat = {
        "messages": [
            {"role": "system", "content": PREFIX},
            {"role": "user", "content": "Hello?"},
        ]
    }
    with TestClient(app=app) as client:
        c = client.post("/v1/chat/completions", json=chat)
        d = client.post("/v1/systemone", json=body())
    assert c.status_code == 200 and c.headers["x-prefix-cache"] == "miss"
    assert c.json()["choices"][0]["message"]["content"] == "ok"
    assert c.json()["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert d.headers["x-prefix-cache"] == "hit"
    assert len(lru(app)) == 1


def test_chat_without_a_generator_is_501():
    with TestClient(app=create_app(FakeRunner())) as client:
        r = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
        )
    assert r.status_code == 501


def test_lru_evicts_least_recently_used():
    built = []
    cache = PrefixLRU(lambda t: built.append(t) or FakeRunner().cache_prefix(t), capacity=2)
    for key in ((1,), (2,), (1,), (3,), (2,)):
        cache.get(key)
    assert built == [(1,), (2,), (3,), (2,)]  # (2,) was evicted by (3,), (1,) kept
    assert (1,) not in cache and (3,) in cache


# --- real model -------------------------------------------------------------------------

_SNAP = HF_HUB / f"models--{GEMMA.replace('/', '--')}" / "snapshots"


@pytest.mark.mlx
@pytest.mark.skipif(
    not any(_SNAP.glob("*/model.safetensors")), reason=f"{GEMMA} weights not in HF cache"
)
def test_real_model_decision_and_chat_share_prefix():
    pytest.importorskip("mlx.core")
    from mlx_classifiers.mlx_runner import MLXRunner

    app = create_app(load=lambda: MLXRunner.load(GEMMA))
    chat = {
        "messages": [{"role": "system", "content": PREFIX}, {"role": "user", "content": "Hi."}],
        "max_tokens": 8,
    }
    with TestClient(app=app) as client:
        first = client.post("/v1/systemone", json=body())
        second = client.post("/v1/systemone", json=body(state="The app crashes on login."))
        c = client.post("/v1/chat/completions", json=chat)
        again = client.post("/v1/systemone", json=body())
    assert first.status_code == 200, first.text
    assert first.json()["usage"]["output_tokens"] == 0
    assert second.headers["x-prefix-cache"] == "hit"
    assert second.json()["usage"]["cached_tokens"] > 0
    assert c.status_code == 200 and c.headers["x-prefix-cache"] == "hit"
    assert c.json()["usage"]["completion_tokens"] > 0
    # Neither route mutates the shared entry: the same request answers the same.
    a, b = first.json()["answers"], again.json()["answers"]
    assert a["department"]["probabilities"] == pytest.approx(b["department"]["probabilities"])
    assert len(app.state.server.prefixes) == 1
