"""Shared fixtures. Real tokenizers load from the local HF cache only (no weights,
no network); tests that need one are skipped when it is not cached."""

from __future__ import annotations

from pathlib import Path

import pytest

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
GEMMA = "mlx-community/gemma-4-E2B-it-qat-4bit"
QWEN = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

_cache: dict[str, object] = {}


def load_cached_tokenizer(repo: str):
    if repo in _cache:
        return _cache[repo]
    snapshots = sorted((HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"))
    snapshots = [s for s in snapshots if (s / "tokenizer.json").exists()]
    if not snapshots:
        pytest.skip(f"{repo} tokenizer not in the HF cache")
    try:
        from mlx_lm.utils import load_tokenizer
    except ImportError:
        pytest.skip("mlx_lm not installed")
    tok = load_tokenizer(snapshots[-1])
    _cache[repo] = tok
    return tok


@pytest.fixture(params=[GEMMA, QWEN], ids=["gemma", "qwen"])
def hf_tokenizer(request):
    return load_cached_tokenizer(request.param)


@pytest.fixture
def gemma_tokenizer():
    return load_cached_tokenizer(GEMMA)


def hf_tokenize(tok):
    return lambda text: tok.encode(text, add_special_tokens=False)
