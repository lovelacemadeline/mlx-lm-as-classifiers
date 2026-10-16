"""Story 2.2 — MLXRunner split prefill, against a tiny fake mlx module (no weights)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx_classifiers import SystemOneRequest, decide
from mlx_classifiers.mlx_runner import MLXRunner
from mlx_classifiers.runner import LogitRow
from mlx_classifiers.template import ToyChatTemplate

VOCAB, DIM = 64, 8


class _SumCache:
    """Running sum of embeddings: hidden at a row depends on every earlier token."""

    def __init__(self):
        self.total = mx.zeros((DIM,))
        self.offset = 0

    @property
    def state(self):
        return self.total


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, DIM)
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, x, cache=None):
        self.calls.append(tuple(x.shape))
        h = self.embed_tokens(x)  # (1, L, D)
        c = cache[0]
        prefix = mx.cumsum(h, axis=1) + c.total
        c.total = prefix[0, -1]
        c.offset += x.shape[1]
        return mx.tanh(prefix)


class _FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(0)
        self.model = _Backbone()
        self.final_logit_softcapping = 30.0

    def make_cache(self):
        return [_SumCache()]


class _ToyTok:
    """Maps toy text to ids < VOCAB: one id per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % VOCAB for ch in text]


def _runner(**kw) -> MLXRunner:
    return MLXRunner(
        model=_FakeLM(),
        tokenizer=_ToyTok(),
        model_id="fake-mlx",
        chat_template=ToyChatTemplate(),
        **kw,
    )


def _full_logits(runner: MLXRunner, tokens, positions):
    """Reference: the whole sequence through backbone + head (what the split avoids)."""
    cache = runner.model.make_cache()
    h = runner.model.model(mx.array(tokens)[None], cache=cache)
    out = runner._head(h[:, mx.array(positions), :])[0].astype(mx.float32)
    runner.head_calls.pop()
    return out


TOKENS = list(range(3, 40))


def test_split_runs_cache_only_rows_then_one_readout_pass():
    r = _runner()
    readout = r.prefill(TOKENS)
    n = len(TOKENS)
    assert r.model.model.calls == [(1, n - 1), (1, 1)]
    assert r.head_calls == [(1, 1, DIM)]  # vocab projection on the readout row only
    assert readout.positions == (n - 1,) and readout.input_tokens == n
    ref = _full_logits(r, TOKENS, [n - 1])
    row = readout.at(n - 1)
    assert isinstance(row, LogitRow) and len(row) == VOCAB
    assert [row[t] for t in range(VOCAB)] == pytest.approx(ref[0].tolist(), abs=1e-5)
    assert row.logsumexp() == pytest.approx(mx.logsumexp(ref[0]).item(), abs=1e-5)


def test_multi_readout_projects_only_gathered_rows():
    r = _runner()
    positions = [10, 12, 20]
    readout = r.prefill(TOKENS, readout_positions=positions)
    # rows [0, 10) cache-only; rows 10..20 one forward; rows after 20 never run
    assert r.model.model.calls == [(1, 10), (1, 11)]
    assert r.head_calls == [(1, 3, DIM)]
    ref = _full_logits(r, TOKENS, positions)
    for i, p in enumerate(positions):
        got = [readout.at(p)[t] for t in range(VOCAB)]
        assert got == pytest.approx(ref[i].tolist(), abs=1e-5)


def test_prefix_cache_is_copied_not_mutated():
    r = _runner()
    cache = r.cache_prefix(TOKENS[:15])
    before = cache.state[0].total.tolist()
    a = r.prefill(TOKENS, prefix_cache=cache)
    b = r.prefill(TOKENS, prefix_cache=cache)
    assert cache.state[0].offset == 15 and cache.state[0].total.tolist() == before
    assert a.input_tokens == len(TOKENS) - 15 and a.cached_tokens == 15
    last = len(TOKENS) - 1
    uncached = r.prefill(TOKENS).at(last)
    for t in (0, 5, 17):
        assert a.at(last)[t] == b.at(last)[t] == pytest.approx(uncached[t], abs=1e-5)


def test_cache_only_prefill_is_chunked():
    r = _runner(prefill_step=8)
    r.cache_prefix(TOKENS[:20])
    assert r.model.model.calls == [(1, 8), (1, 8), (1, 4)]


def test_decide_through_mlx_runner_on_fake_module():
    req = SystemOneRequest.model_validate(
        {
            "prefix": "rules",
            "state": "s",
            "questions": {
                "q": {"type": "choice", "instructions": "?", "criteria": {"x": None, "y": None}}
            },
        }
    )
    r = _runner()
    resp = decide(req, r)
    assert resp.usage.output_tokens == 0 and resp.usage.cached_tokens > 0
    assert resp.timing_ms.prefill > 0 and resp.timing_ms.total >= resp.timing_ms.prefill
    assert r.head_calls == [(1, 1, DIM)]
