"""Story 3.1 — /v1/systemone beside /v1/chat/completions, one process, one prefix LRU."""

from __future__ import annotations

import pytest
from conftest import GEMMA, HF_HUB

pytest.importorskip("litestar")
from litestar.testing import TestClient

from mlx_classifiers.prompt_cache import PromptCacheStore
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


# --- story 3.5: max_tokens, finish_reason, usage, per-call thinking ----------------------


class ScriptedGenerate:
    """A generator over the FakeRunner that emits ``script``'s tokens, cut at
    ``max_tokens`` (``length``) or whole (``stop``), and records every call."""

    def __init__(self, script: str = "The answer is four."):
        self.script = script
        self.calls: list[dict] = []

    def __call__(self, runner, tokens, prefix_cache, max_tokens, temperature):
        self.calls.append({"tokens": list(tokens), "cache": prefix_cache, "max_tokens": max_tokens})
        ids = runner.tokenize(self.script)
        finish = "length" if len(ids) > max_tokens else "stop"
        ids = ids[:max_tokens]
        text = runner.tokenizer.decode(ids)
        return ChatResult(text, len(ids) + (finish == "stop"), finish, tuple(ids))


def chat_body(**extra):
    msgs = [{"role": "system", "content": PREFIX}, {"role": "user", "content": "Plan it."}]
    return {"messages": msgs, **extra}


def test_chat_honours_max_tokens_with_finish_reason_and_usage():
    gen = ScriptedGenerate("one two three four five six seven eight")
    app = create_app(FakeRunner(), generate=gen, max_tokens_ceiling=4096)
    n_script = len(FakeRunner().tokenize(gen.script))
    with TestClient(app=app) as client:
        default = client.post("/v1/chat/completions", json=chat_body())
        short = client.post("/v1/chat/completions", json=chat_body(max_tokens=3))
        big = client.post("/v1/chat/completions", json=chat_body(max_completion_tokens=4096))
        over = client.post("/v1/chat/completions", json=chat_body(max_tokens=4097))
        bad = client.post("/v1/chat/completions", json=chat_body(max_tokens=0))
    assert [c["max_tokens"] for c in gen.calls] == [512, 3, 4096]  # never capped below
    d, sh = default.json(), short.json()
    assert d["choices"][0]["finish_reason"] == "stop"
    assert sh["choices"][0]["finish_reason"] == "length"
    assert sh["usage"]["completion_tokens"] == 3
    assert d["usage"]["completion_tokens"] == n_script + 1  # plus the stop token
    prompt = FakeRunner().tokenize(
        FakeRunner().chat_template.render_messages(chat_body()["messages"]).text
    )
    assert d["usage"]["prompt_tokens"] == len(prompt) == len(gen.calls[0]["tokens"])
    assert d["usage"]["total_tokens"] == len(prompt) + n_script + 1
    assert d["usage"]["completion_tokens_details"] == {"reasoning_tokens": 0}
    assert "reasoning" not in d["choices"][0]["message"]
    assert big.status_code == 200
    assert over.status_code == 422 and "4096" in over.json()["detail"]
    assert over.json()["errors"][0]["loc"] == "max_tokens"
    assert bad.status_code == 422


def test_enable_thinking_renders_a_different_prompt_for_that_call_only():
    gen = ScriptedGenerate()
    app = create_app(FakeRunner(), generate=gen)
    with TestClient(app=app) as client:
        off = client.post("/v1/chat/completions", json=chat_body())
        on = client.post("/v1/chat/completions", json=chat_body(enable_thinking=True))
        off_again = client.post("/v1/chat/completions", json=chat_body(enable_thinking=False))
        bad = client.post("/v1/chat/completions", json=chat_body(enable_thinking="yes"))
    assert off.status_code == on.status_code == off_again.status_code == 200
    plain, thinking, plain_again = (c["tokens"] for c in gen.calls)
    assert thinking != plain and plain_again == plain
    runner = FakeRunner()
    assert (
        runner.tokenize("<|think|>")[0] in thinking and runner.tokenize("<|think|>")[0] not in plain
    )
    assert bad.status_code == 422 and bad.json()["errors"][0]["loc"] == "enable_thinking"
    # Decisions never render in thinking mode, whatever chats asked for.
    rendered = runner.chat_template.render(PREFIX, "state")
    assert "<|think|>" not in rendered.text


def test_thinking_trace_is_split_into_reasoning():
    trace = "<|channel|>thought\nTwo plus two.<|channelend|>Four."
    gen = ScriptedGenerate(trace)
    app = create_app(FakeRunner(), generate=gen)
    n_span = len(FakeRunner().tokenize("<|channel|>thought\nTwo plus two.<|channelend|>"))
    with TestClient(app=app) as client:
        r = client.post("/v1/chat/completions", json=chat_body(enable_thinking=True))
        cut = client.post(
            "/v1/chat/completions", json=chat_body(enable_thinking=True, max_tokens=5)
        )
    msg = r.json()["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "Four.", "reasoning": "Two plus two."}
    assert r.json()["usage"]["completion_tokens_details"]["reasoning_tokens"] == n_span
    c = cut.json()
    assert c["choices"][0]["finish_reason"] == "length"
    assert c["choices"][0]["message"]["content"] == ""  # cut off inside the trace
    assert c["choices"][0]["message"]["reasoning"] == "Two"
    assert c["usage"]["completion_tokens_details"]["reasoning_tokens"] == 5


@pytest.mark.parametrize("thinking", [False, True], ids=["plain", "thinking"])
def test_long_chat_leaves_the_decision_prefix_entry_untouched(thinking):
    """A long planning call runs on its own KV: the decision prefix entry is the
    same object afterwards, still hits, and answers identically."""
    gen = ScriptedGenerate(" ".join(["step"] * 3000))
    app = create_app(FakeRunner(), generate=gen, prefix_cache_size=1)
    with TestClient(app=app) as client:
        before = client.post("/v1/systemone", json=body())
        ((key, entry),) = lru(app)._entries.items()
        chat = client.post(
            "/v1/chat/completions",
            json=chat_body(max_tokens=2048, enable_thinking=thinking),
        )
        after = client.post("/v1/systemone", json=body())
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["finish_reason"] == "length"
    assert chat.json()["usage"]["completion_tokens"] == 2048
    # thinking renders a different system turn: it peeks (misses) and inserts nothing
    assert chat.headers["x-prefix-cache"] == ("miss" if thinking else "hit")
    assert (gen.calls[0]["cache"] is None) == thinking
    assert lru(app)._entries == {key: entry} and lru(app)._entries[key] is entry
    assert after.headers["x-prefix-cache"] == "hit"
    assert after.json()["answers"] == before.json()["answers"]


def test_thinking_chat_reads_a_matching_prefix_without_writing_the_lru():
    """When thinking leaves the system turn alone (e.g. Qwen3), the call shares
    the decision's entry read-only."""

    class SameSystemTurn(type(FakeRunner().chat_template)):
        def render_messages(self, messages, *, enable_thinking=False):
            return super().render_messages(messages)

    gen = ScriptedGenerate()
    app = create_app(FakeRunner(chat_template=SameSystemTurn()), generate=gen)
    with TestClient(app=app) as client:
        client.post("/v1/systemone", json=body())
        hits = (lru(app).hits, lru(app).misses)
        r = client.post("/v1/chat/completions", json=chat_body(enable_thinking=True))
    assert r.headers["x-prefix-cache"] == "hit"
    assert gen.calls[0]["cache"] is next(iter(lru(app)._entries.values()))
    assert (lru(app).hits, lru(app).misses) == hits  # peek counts nothing


# --- story 3.4: delta prefix caching -----------------------------------------------------------------------------

GRID_QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "Which way?",
        "criteria": {"up": None, "down": None, "left": None, "right": None},
    },
    "safe": {"type": "noul", "instructions": "Is the agent safe?"},
}


def _body(state, **options):
    b = {"prefix": "Rules of the grid.", "state": state, "questions": GRID_QUESTIONS}
    if options:
        b["options"] = options
    return b


def test_server_reports_the_delta_hit_length():
    state = "Map: " + "wall floor " * 40
    app = create_app(FakeRunner())
    assert isinstance(app.state.server.runner.prompt_cache, PromptCacheStore)
    with TestClient(app=app) as client:
        first = client.post("/v1/systemone", json=_body(state + "agent at 1 1."))
        second = client.post("/v1/systemone", json=_body(state + "agent at 1 2."))
        off = client.post("/v1/systemone", json=_body(state + "agent at 1 3.", delta_cache=False))
    prefix = int(first.headers["x-prefix-cache-tokens"])
    assert first.headers["x-prefix-cache"] == "miss" and prefix > 0  # declared prefix only
    assert second.headers["x-prefix-cache"] == "hit"
    hit = int(second.headers["x-prefix-cache-tokens"])
    assert hit > prefix + 80  # the map was reused, not just the rules
    assert second.json()["usage"]["cached_tokens"] >= hit
    assert int(off.headers["x-prefix-cache-tokens"]) == prefix


def test_server_flag_turns_the_delta_cache_off():
    runner = FakeRunner(prompt_cache=PromptCacheStore())
    app = create_app(runner, delta_cache=False)
    assert runner.prompt_cache is None
    state = "Map: " + "wall floor " * 40
    with TestClient(app=app) as client:
        client.post("/v1/systemone", json=_body(state + "a."))
        second = client.post("/v1/systemone", json=_body(state + "b."))
    assert second.headers["x-prefix-cache"] == "hit"
    assert int(second.headers["x-prefix-cache-tokens"]) < 20  # the declared prefix only


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


@pytest.mark.mlx
@pytest.mark.skipif(
    not any(_SNAP.glob("*/model.safetensors")), reason=f"{GEMMA} weights not in HF cache"
)
def test_real_model_thinking_plan_call(capsys):
    """Story 3.5 on E2B: a 2048-token-cap thinking completion either returns a
    trace or runs to the cap; a decision around it is unchanged and still hits."""
    import time

    pytest.importorskip("mlx.core")
    from mlx_classifiers.mlx_runner import MLXRunner

    app = create_app(load=lambda: MLXRunner.load(GEMMA))
    plan = {
        "messages": [
            {"role": "system", "content": PREFIX},
            {
                "role": "user",
                "content": "Plan, step by step, how to triage a week-long payout outage.",
            },
        ],
        "max_tokens": 2048,
        "enable_thinking": True,
    }
    with TestClient(app=app, timeout=600) as client:
        before = client.post("/v1/systemone", json=body())
        t0 = time.perf_counter()
        c = client.post("/v1/chat/completions", json=plan)
        wall = time.perf_counter() - t0
        after = client.post("/v1/systemone", json=body())
    assert c.status_code == 200, c.text
    out = c.json()
    choice, usage = out["choices"][0], out["usage"]
    with capsys.disabled():
        print(
            f"\n[3.5] finish={choice['finish_reason']} wall={wall:.1f}s "
            f"completion={usage['completion_tokens']} "
            f"reasoning_tokens={usage['completion_tokens_details']['reasoning_tokens']} "
            f"reasoning_chars={len(choice['message'].get('reasoning') or '')} "
            f"content_chars={len(choice['message']['content'])}"
        )
    assert choice["message"].get("reasoning") or choice["finish_reason"] == "length"
    assert usage["completion_tokens"] <= 2048
    assert after.headers["x-prefix-cache"] == "hit"
    a, b = before.json()["answers"], after.json()["answers"]
    assert a["has_repro"]["noul"] == pytest.approx(b["has_repro"]["noul"])
    for q in ("department", "severity"):
        assert a[q]["probabilities"] == pytest.approx(b[q]["probabilities"])
