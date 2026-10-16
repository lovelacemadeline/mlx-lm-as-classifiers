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
    uncached = r.prefill(TOKENS, delta=False).at(last)
    for t in (0, 5, 17):
        assert a.at(last)[t] == b.at(last)[t] == pytest.approx(uncached[t], abs=1e-5)


def test_cache_only_prefill_is_chunked():
    r = _runner(prefill_step=8)
    r.cache_prefix(TOKENS[:20])
    assert r.model.model.calls == [(1, 4), (1, 8), (1, 8)]  # remainder first (story 3.4)


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


# --- story 3.4: delta prefix caching on real mlx-lm cache classes ------------------------

from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache

from mlx_classifiers.mlx_runner import rewind_cache
from mlx_classifiers.prompt_cache import PromptCacheStore
from mlx_classifiers.runner import PrefixCache

WINDOW = 8


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(DIM, DIM, bias=False) for _ in range(3))
        self.rope = nn.RoPE(DIM)

    def __call__(self, h, mask, cache):
        B, L, _ = h.shape

        def heads(x):
            return x.reshape(B, L, 1, DIM).transpose(0, 2, 1, 3)

        q = self.rope(heads(self.q(h)), offset=cache.offset)
        k = self.rope(heads(self.k(h)), offset=cache.offset)
        k, v = cache.update_and_fetch(k, heads(self.v(h)))
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=DIM**-0.5, mask=mask)
        return out.transpose(0, 2, 1, 3).reshape(B, L, DIM)


class _AttnBackbone(nn.Module):
    """One sliding-window layer (RotatingKVCache, window 8) and one global layer
    (KVCache), masks built the way Gemma builds them: a toy with mlx-lm's real
    cache classes, so rewinds are tested against their update paths."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, DIM)
        self.layers = [_Attn(), _Attn()]
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, x, cache=None):
        self.calls.append(tuple(x.shape))
        h = self.embed_tokens(x)
        masks = [
            create_attention_mask(h, cache[0], window_size=WINDOW),
            create_attention_mask(h, cache[1]),
        ]
        for layer, c, m in zip(self.layers, cache, masks, strict=True):
            h = h + layer(h, m, c)
        return mx.tanh(h)


class _AttnLM(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(1)
        self.model = _AttnBackbone()

    def make_cache(self):
        return [RotatingKVCache(max_size=WINDOW, keep=0), KVCache()]


def _attn_runner(**kw) -> MLXRunner:
    return MLXRunner(
        model=_AttnLM(),
        tokenizer=_ToyTok(),
        model_id="fake-attn",
        chat_template=ToyChatTemplate(),
        **kw,
    )


def _logits(readout, pos):
    return [readout.at(pos)[t] for t in range(VOCAB)]


SEQ = [(7 * i + 3) % VOCAB for i in range(40)]


def _variant(base, at, n_new=None, seed=0):
    """``base[:at]`` then fresh tokens up to ``len(base)`` (differing at ``at``)."""
    n_new = len(base) - at if n_new is None else n_new
    tail = [(base[at] + 1 + 5 * i + seed) % VOCAB for i in range(n_new)]
    return list(base[:at]) + tail


def test_delta_prefill_computes_only_the_delta_and_matches_uncached():
    r = _attn_runner()
    r.prefill(SEQ)
    b = _variant(SEQ, 30)  # shares 30 tokens, well past the window of 8
    r.model.model.calls.clear()
    got = r.prefill(b)
    assert got.cached_tokens == 30 and got.input_tokens == len(b) - 30
    assert r.prefill_calls[-1] == (30, len(b) - 30)
    assert r.model.model.calls == [(1, len(b) - 31), (1, 1)]  # only the delta, then readout
    ref = r.prefill(b, delta=False)
    assert ref.cached_tokens == 0
    assert _logits(got, len(b) - 1) == pytest.approx(_logits(ref, len(b) - 1), abs=1e-4)


def test_rewind_after_the_ring_slid_is_exact_or_refused():
    """Entry Y was itself built on a rewind, so its sliding layer holds only
    W - 1 rows plus its own delta: Y can rewind into that delta, not before it."""
    r = _attn_runner()
    r.prompt_cache = PromptCacheStore(max_entries=1)  # X is evicted once Y is stored
    x = SEQ
    y = _variant(x, 35)
    r.prefill(x)
    assert r.prefill(y).cached_tokens == 35
    (key,) = r.prompt_cache.keys()
    state = r.prompt_cache._entries[key].state
    assert key == tuple(y[:-1])
    ring = state[0]
    assert ring.offset == 39 and ring.offset - ring.keys.shape[2] > 0  # it slid

    assert rewind_cache(state, 39, 37) is not None  # inside Y's own delta
    assert rewind_cache(state, 39, 33) is None  # needs rows the window dropped
    assert state[0].offset == 39 and ring.keys.shape[2] == 11  # never mutated

    z = _variant(y, 37, seed=3)  # representable from Y
    got = r.prefill(z)
    assert got.cached_tokens == 37
    ref = r.prefill(z, delta=False)
    assert _logits(got, len(z) - 1) == pytest.approx(_logits(ref, len(z) - 1), abs=1e-4)

    q = _variant(z, 33, seed=7)  # only Z is stored; it cannot rewind to 33
    before = r.prompt_cache.unrepresentable
    got = r.prefill(q)
    assert got.cached_tokens == 0 and r.prompt_cache.unrepresentable == before + 1
    ref = r.prefill(q, delta=False)
    assert _logits(got, len(q) - 1) == pytest.approx(_logits(ref, len(q) - 1), abs=1e-4)


def test_hit_within_the_window_and_declared_prefix_fallback():
    r = _attn_runner()
    r.prefill(SEQ)
    short = _variant(SEQ, 5)  # hit of 5 tokens, inside the window
    got = r.prefill(short)
    assert got.cached_tokens == 5
    ref = r.prefill(short, delta=False)
    assert _logits(got, len(short) - 1) == pytest.approx(_logits(ref, len(short) - 1), abs=1e-4)
    # A declared prefix longer than any delta hit wins; the delta hit wins when longer.
    r2 = _attn_runner()
    pc = r2.cache_prefix(SEQ[:20], delta=False)
    assert r2.prefill(SEQ, pc).cached_tokens == 20
    assert r2.prefill(_variant(SEQ, 36), pc).cached_tokens == 36


def test_ring_buffer_rewind_by_one_continues_exactly():
    """A cache advanced one token at a time (the in-place ring path, as decode
    leaves it) can drop exactly its newest token, and continues correctly."""
    r = _attn_runner()
    cache = r.model.make_cache()
    for t in SEQ[:12]:
        r._backbone(mx.array([[t]]), cache=cache)
        mx.eval([c.state for c in cache])
    assert cache[0].offset == 12 and cache[0]._idx < cache[0].offset  # wrapped ring
    assert rewind_cache(cache, 12, 10) is None
    back = rewind_cache(cache, 12, 11)
    got = r.prefill(SEQ[:20], PrefixCache(tuple(SEQ[:11]), back), delta=False)
    ref = r.prefill(SEQ[:20], delta=False)
    assert _logits(got, 19) == pytest.approx(_logits(ref, 19), abs=1e-4)


def test_chunked_prefill_keeps_the_last_chunk_full_for_rewinds():
    """39 cache-only rows at prefill_step 16 run as 7 + 16 + 16, so the sliding
    layer keeps W - 1 + 16 rows and a later rewind of 9 is representable (with
    16 + 16 + 7 it would keep only W - 1 + 7)."""
    r = _attn_runner()
    r.prefill_step = 16
    r.prefill(SEQ)
    assert r.model.model.calls[:3] == [(1, 7), (1, 16), (1, 16)]
    b = _variant(SEQ, 30)
    got = r.prefill(b)
    assert got.cached_tokens == 30
    ref = r.prefill(b, delta=False)
    assert _logits(got, len(b) - 1) == pytest.approx(_logits(ref, len(b) - 1), abs=1e-4)


def test_delta_off_neither_reads_nor_writes_the_store():
    r = _attn_runner()
    r.prefill(SEQ, delta=False)
    assert len(r.prompt_cache) == 0
    r.prefill(SEQ)
    assert r.prefill(SEQ, delta=False).cached_tokens == 0
    r.prompt_cache = None
    assert r.prefill(SEQ).cached_tokens == 0
