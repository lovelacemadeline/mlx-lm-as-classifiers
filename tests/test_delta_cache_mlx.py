"""Story 3.4 — delta prefix caching on the real Gemma 4 E2B (sliding window 512)."""

from __future__ import annotations

import statistics
import time

import pytest
from conftest import GEMMA, HF_HUB

mx = pytest.importorskip("mlx.core")

import itertools

from mlx_classifiers import SystemOneRequest, decide
from mlx_classifiers.bench import PREFIX, QUESTION, delta_states, filler_state
from mlx_classifiers.fixtures import DEFAULT_PROB_TOL
from mlx_classifiers.mlx_runner import MLXRunner
from mlx_classifiers.prompt_cache import PromptCacheStore

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


@pytest.fixture
def fresh(runner):
    runner.prompt_cache = PromptCacheStore()
    runner.prefill_calls.clear()
    yield runner
    runner.prompt_cache = PromptCacheStore()


def _req(state, **options):
    body = {"prefix": PREFIX, "state": state, "questions": {"q": QUESTION}}
    if options:
        body["options"] = options
    return SystemOneRequest.model_validate(body)


def _probs(resp):
    return list(resp.answers["q"].raw_probabilities.values())


def _assert_same(a, b, tol=DEFAULT_PROB_TOL):
    pa, pb = _probs(a), _probs(b)
    assert max(abs(x - y) for x, y in zip(pa, pb, strict=True)) <= tol, (pa, pb)
    assert a.answers["q"].choice == b.answers["q"].choice


def test_1024_state_with_a_50_token_tail_change_matches_uncached(fresh):
    s0, s1 = delta_states(fresh.tokenize, 1024, 50, 2)
    decide(_req(s0), fresh)
    got = decide(_req(s1), fresh)
    ref = decide(_req(s1, delta_cache=False), fresh)
    n_hit = fresh.prefill_calls[1][0]
    assert n_hit > 900  # rules + ~970 state tokens reused, past the 512 window
    assert got.usage.cached_tokens == n_hit
    assert got.usage.input_tokens < 120 < ref.usage.input_tokens
    _assert_same(got, ref)


def test_sliding_layers_rewound_past_the_window_match_uncached(fresh):
    """Three ticks whose change points move earlier. Tick 2 rewinds tick 1's
    fully retained entry to ~850 tokens (past the 512 window). Tick 3 changes
    earlier than tick 2 did, which tick 2's sliding layers cannot rewind to (they
    kept 511 rows plus their own delta): with tick 1 still stored the lookup
    falls back to it; with only tick 2 stored, to the declared prefix. Every
    answer matches the uncached path."""
    head, _ = filler_state(fresh.tokenize, 800)
    words = head.split()
    ticks = [
        head + " The agent is at row one.",
        " ".join(words[:-20] + ["pending"] + words[-19:]) + " The agent is at row two.",
        " ".join(words[:-60] + ["blocked"] + words[-59:]) + " The agent is at row three.",
    ]
    for entries in (8, 1):
        fresh.prompt_cache = PromptCacheStore(max_entries=entries)
        hits = []
        for t in ticks:
            got = decide(_req(t), fresh)
            hits.append(fresh.prefill_calls[-1][0])
            _assert_same(got, decide(_req(t, delta_cache=False), fresh))
        prefix_len = hits[0]
        assert hits[1] > 800
        if entries == 8:  # tick 1's entry serves tick 3
            assert 512 < hits[2] < hits[1] and fresh.prompt_cache.unrepresentable == 0
        else:  # tick 2's entry is refused; only the declared prefix is left
            assert hits[2] == prefix_len and fresh.prompt_cache.unrepresentable == 1


def test_lru_bounds_entries_and_bytes_on_real_kv(fresh):
    states = delta_states(fresh.tokenize, 600, 50, 4)
    fresh.prompt_cache = PromptCacheStore(max_entries=2)
    for s in states:
        decide(_req(s), fresh)
    assert len(fresh.prompt_cache) == 2
    one = fresh.prompt_cache.nbytes // 2
    fresh.prompt_cache = PromptCacheStore(max_bytes=int(one * 1.5))
    for s in states:
        decide(_req(s), fresh)
    assert len(fresh.prompt_cache) == 1 and fresh.prompt_cache.nbytes <= int(one * 1.5)


def test_transcript_append_only_20_steps_grows_cached_and_stays_flat(fresh, capsys):
    """Each step appends ~100 tokens: cached_tokens grows by about that much and
    the per-step time does not grow with the transcript."""
    turn, _ = filler_state(fresh.tokenize, 95)
    transcript, cached, ms = "", [], []
    for step in range(20):
        transcript += f"\nStep {step}: " + turn
        t = time.perf_counter()
        resp = decide(_req(transcript), fresh)
        ms.append((time.perf_counter() - t) * 1e3)
        cached.append(resp.usage.cached_tokens)
    with capsys.disabled():
        print(f"\n[3.4 transcript] cached={cached}\n[3.4 transcript] ms={[round(x) for x in ms]}")
    assert all(b > a + 80 for a, b in itertools.pairwise(cached))
    early, late = statistics.median(ms[2:7]), statistics.median(ms[15:20])
    assert late < 2.0 * early, (early, late)
