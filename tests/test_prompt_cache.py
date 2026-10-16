"""Story 3.4 — delta prefix caching: the store, the FakeRunner, decide and the server."""

from __future__ import annotations

import itertools

import pytest

from mlx_classifiers import SystemOneRequest, decide
from mlx_classifiers.prompt_cache import PromptCacheStore, common_prefix_len
from mlx_classifiers.runner import FakeRunner


def _rewind(state, stored_len, hit_len):
    return ("rewound", state, hit_len)


def test_common_prefix_len():
    assert common_prefix_len((1, 2, 3), (1, 2, 4)) == 2
    assert common_prefix_len((1, 2), (1, 2, 3)) == 2
    assert common_prefix_len((), (1,)) == 0
    assert common_prefix_len(tuple(range(1000)), tuple(range(999)) + (5,)) == 999


def test_lookup_returns_the_longest_common_prefix_capped():
    s = PromptCacheStore()
    s.insert((1, 2, 3, 4), "a", 1)
    s.insert((1, 2, 9), "b", 1)
    assert s.lookup((1, 2, 3, 7, 7), _rewind) == (3, ("rewound", "a", 3))
    assert s.lookup((1, 2, 3, 4, 5), _rewind, max_len=2)[0] == 2  # a and b tie at 2
    assert s.lookup((1, 2, 3, 4, 5), _rewind, min_len=4) == (0, None)  # nothing better
    assert s.lookup((5,), _rewind) == (0, None)
    assert (s.hits, s.misses) == (2, 2)


def test_lookup_falls_back_when_a_rewind_is_not_representable():
    s = PromptCacheStore()
    s.insert((1, 2, 3, 4, 5), "deep", 1)
    s.insert((1, 2, 3), "shallow", 1)

    def only_shallow(state, stored_len, hit_len):
        return None if state == "deep" else state

    assert s.lookup((1, 2, 3, 4, 9), only_shallow) == (3, "shallow")
    assert s.unrepresentable == 1
    assert s.lookup((1, 2, 3, 4, 9), lambda *a: None) == (0, None)


def test_lru_evicts_by_entries_and_by_bytes():
    s = PromptCacheStore(max_entries=2, max_bytes=100)
    s.insert((1,), "a", 10)
    s.insert((2,), "b", 10)
    s.lookup((1, 5), _rewind)  # bumps (1,)
    s.insert((3,), "c", 10)
    assert s.keys() == [(1,), (3,)]  # (2,) was least recently used
    assert s.insert((3,), "c2", 10) is False and s.keys() == [(1,), (3,)]
    s.insert((4,), "d", 95)  # over the byte budget: evicts until it fits
    assert s.keys() == [(4,)] and s.nbytes == 95
    assert s.insert((5,), "huge", 101) is False and (5,) not in s
    with pytest.raises(ValueError):
        PromptCacheStore(max_entries=0)


def test_fake_runner_prefills_only_the_delta():
    r = FakeRunner(prompt_cache=PromptCacheStore())
    a = list(range(10, 60))
    b = a[:44] + [1, 2, 3, 4, 5, 6]
    first = r.prefill(a)
    second = r.prefill(b)
    assert first.cached_tokens == 0 and r.prefill_calls[0] == (0, 50)
    assert second.cached_tokens == 44 and r.prefill_calls[1] == (44, 6)
    plain = FakeRunner().prefill(b)
    ids = range(0, 50_000, 4999)
    assert [second.at(49)[i] for i in ids] == [plain.at(49)[i] for i in ids]
    assert r.prefill(b, delta=False).cached_tokens == 0


QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "Which way?",
        "criteria": {"up": None, "down": None, "left": None, "right": None},
    },
    "safe": {"type": "noul", "instructions": "Is the agent safe?"},
}


def _req(state, **options):
    body = {"prefix": "Rules of the grid.", "state": state, "questions": QUESTIONS}
    if options:
        body["options"] = options
    return SystemOneRequest.model_validate(body)


def test_decide_reuses_the_shared_head_across_questions_and_permutations():
    state = "Map: " + "wall floor " * 60 + "agent at 3 4."
    off = FakeRunner()
    on = FakeRunner(prompt_cache=PromptCacheStore())
    base = decide(_req(state, permutations=3), off)
    got = decide(_req(state, permutations=3), on)
    assert got.answers == base.answers  # same tokens, same logits
    assert got.usage.cached_tokens > base.usage.cached_tokens
    assert got.usage.input_tokens < base.usage.input_tokens
    # every prompt after the first reuses more than the declared prefix
    prefix_len = on.prefill_calls[0][0]
    assert all(c > prefix_len for c, _ in on.prefill_calls[1:])


def test_options_delta_cache_false_runs_without_the_store():
    r = FakeRunner(prompt_cache=PromptCacheStore())
    state = "Map: " + "wall " * 50
    decide(_req(state, delta_cache=False), r)
    assert len(r.prompt_cache) == 0
    first = decide(_req(state), r)
    again = decide(_req(state + " moved", delta_cache=False), r)
    later = decide(_req(state + " moved"), r)
    assert again.usage.cached_tokens < later.usage.cached_tokens
    assert first.answers["move"].probabilities is not None


def test_transcript_steps_report_growing_cached_tokens():
    """Append-only transcript: each step caches everything before its new turn."""
    r = FakeRunner(prompt_cache=PromptCacheStore())
    transcript, cached, computed = "", [], []
    for step in range(20):
        transcript += f" Step {step}: " + "the agent moved east past the wall " * 12
        resp = decide(_req(transcript, readout="first-token"), r)
        n = len(r.prefill_calls)
        cached.append(resp.usage.cached_tokens)
        computed.append(sum(c for _, c in r.prefill_calls[n - 2 :]))
    assert all(b > a for a, b in itertools.pairwise(cached))
    assert max(computed[1:]) < 2 * min(computed[1:])  # per-step work stays flat
