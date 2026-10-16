"""Story 1.2 — Runner protocol and the deterministic FakeRunner (ADR-0002)."""

from __future__ import annotations

import string

import pytest

from mlx_classifiers.runner import FakeRunner, PrefixCache, Readout, Runner, ToyTokenizer


def _row(readout: Readout, pos: int, ids) -> list[float]:
    return [readout.at(pos)[i] for i in ids]


def test_fake_runner_satisfies_runner_protocol():
    runner: Runner = FakeRunner()  # static check; and a runtime smoke of the three methods
    toks = runner.tokenize("hello world")
    cache = runner.cache_prefix(toks[:1])
    assert isinstance(runner.prefill(toks, cache), Readout)


def test_prefill_is_deterministic():
    toks = FakeRunner().tokenize("<|bos|><|user|>\nIs this a bug?<|end|>\n<|assistant|>\n")
    ids = range(0, 50_000, 997)
    a = _row(FakeRunner().prefill(toks), len(toks) - 1, ids)
    b = _row(FakeRunner().prefill(toks), len(toks) - 1, ids)
    assert a == b
    other = _row(FakeRunner().prefill(toks[:-1]), len(toks) - 2, ids)
    assert other != a  # logits depend on the context


def test_prefix_cache_counts_only_suffix_tokens():
    runner = FakeRunner()
    prefix = runner.tokenize("<|bos|><|system|>\nYou route tickets.<|end|>\n")
    suffix = runner.tokenize("<|user|>\nPayouts fail.<|end|>\n<|assistant|>\n")
    tokens = prefix + suffix
    cache = runner.cache_prefix(prefix)
    readout = runner.prefill(tokens, prefix_cache=cache)
    assert readout.input_tokens == len(suffix)
    assert readout.cached_tokens == len(prefix) == len(cache)
    uncached = runner.prefill(tokens)
    assert uncached.input_tokens == len(tokens) and uncached.cached_tokens == 0
    ids = range(0, 50_000, 1013)
    assert _row(readout, len(tokens) - 1, ids) == _row(uncached, len(tokens) - 1, ids)


def test_logits_only_at_readout_positions():
    runner = FakeRunner()
    tokens = runner.tokenize("one two three four five six")
    readout = runner.prefill(tokens, readout_positions=[2, 5])
    assert readout.positions == (2, 5)
    assert len(readout.logits) == 2
    assert runner.projected_rows == 2  # no full-sequence vocab projection
    with pytest.raises(KeyError):
        readout.at(3)
    default = runner.prefill(tokens)
    assert default.positions == (len(tokens) - 1,)
    assert len(default.at(len(tokens) - 1)) == runner.vocab_size


def test_prefill_rejects_bad_cache_and_positions():
    runner = FakeRunner()
    tokens = runner.tokenize("alpha beta gamma delta")
    with pytest.raises(ValueError, match="prefix cache"):
        runner.prefill(tokens, PrefixCache(tokens=(1, 2, 3)))
    cache = runner.cache_prefix(tokens[:2])
    with pytest.raises(ValueError, match="outside the prefilled suffix"):
        runner.prefill(tokens, cache, readout_positions=[1])
    with pytest.raises(ValueError, match="nothing to prefill"):
        runner.prefill(tokens[:2], cache)


def test_logit_bias_shifts_named_tokens():
    tok = ToyTokenizer()
    (yes,) = tok.encode("Yes")
    tokens = tok.encode("<|assistant|>\n")
    base = FakeRunner().prefill(tokens).at(len(tokens) - 1)[yes]
    biased = FakeRunner(logit_bias={yes: 10.0}).prefill(tokens).at(len(tokens) - 1)[yes]
    assert biased == pytest.approx(base + 10.0)


def test_toy_tokenizer_is_stable_and_subword():
    tok = ToyTokenizer()
    assert tok.encode("billing") == ToyTokenizer().encode("billing")
    assert tok.pieces(" billing") == [" bill", "ing"]
    assert tok.pieces("<|bos|>A. yes\n\n") == ["<|bos|>", "A", ".", " yes", "\n\n"]
    letters = [tok.encode(c) for c in string.ascii_uppercase]
    assert all(len(ids) == 1 for ids in letters)
    assert len({ids[0] for ids in letters}) == 26  # no hash collisions among labels
    assert tok.encode("A") != tok.encode(" A")
    assert tok.decode(tok.encode("<|user|>\nhello there")) == "<|user|>\nhello there"
