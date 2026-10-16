"""Epic 4 — golden fixtures: format + generator (4.1) and checker (4.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import GEMMA, HF_HUB

from mlx_classifiers import FakeRunner
from mlx_classifiers.fixtures import (
    FIXTURE_FORMAT,
    SEED_CASES,
    FixtureCase,
    build_fixture,
    generate,
    load_fixtures,
    run_case,
)

COMMITTED = Path(__file__).resolve().parent.parent / "fixtures"
_SNAP = HF_HUB / f"models--{GEMMA.replace('/', '--')}" / "snapshots"
_HAVE_GEMMA = any(_SNAP.glob("*/model.safetensors"))


# --- 4.1: format and generator --------------------------------------------------------------


def test_generate_twice_is_byte_identical(tmp_path):
    a = generate(tmp_path / "a", FakeRunner())
    b = generate(tmp_path / "b", FakeRunner())
    assert [p.name for p in a] == [p.name for p in b] == [f"{c.id}.json" for c in SEED_CASES]
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_bytes() == pb.read_bytes()


def test_seed_set_covers_types_modes_cache_state_shapes_and_an_error():
    def kind(c):
        return next(iter(c.request["questions"].values()))["type"]

    def mode(c):
        return c.request.get("options", {}).get("readout", "first-token")

    combos = {(kind(c), mode(c)) for c in SEED_CASES if c.label_style == "letters"}
    assert combos == {
        (k, m) for k in ("noul", "choice", "score") for m in ("first-token", "sequence")
    }
    assert any(c.prefix_cached for c in SEED_CASES)
    assert any(isinstance(c.request["state"], dict) for c in SEED_CASES)
    assert any(isinstance(c.request["state"], list) for c in SEED_CASES)
    assert any(c.label_style == "option-text" for c in SEED_CASES)
    assert len({c.id for c in SEED_CASES}) == len(SEED_CASES)


def test_fixture_pins_provenance_path_and_readout():
    fx = build_fixture(
        SEED_CASES[0], FakeRunner(), {"repo": "fake", "revision": None, "tokenizer_sha256": None}
    )
    assert fx["format"] == FIXTURE_FORMAT and fx["id"] == SEED_CASES[0].id
    assert fx["path"]["prefill"] == "split"
    assert fx["readout"] == "first-token" and fx["question"] == "q"
    e = fx["expected"]
    assert e["readout_position"] == len(e["tokens"]) - 1
    assert set(e["option_token_ids"]) == set(e["options"]) == {"yes", "no"}
    assert sum(e["raw_probabilities"].values()) == pytest.approx(1.0)
    assert e["answer"] == max(e["raw_probabilities"], key=e["raw_probabilities"].get)


def test_cached_case_reuses_the_system_turn():
    runner = FakeRunner()
    case = next(c for c in SEED_CASES if c.prefix_cached)
    before = runner.prefilled_tokens
    out = run_case(case, runner)
    n, k = len(out["tokens"]), out["prefix_len"]
    assert k > 0 and runner.prefilled_tokens - before == n  # k cached once + (n - k) suffix
    uncached = run_case(FixtureCase("u", "", case.request), FakeRunner())
    assert uncached["raw_logits"] == out["raw_logits"]  # the fake is exact; MLX is per-path


def test_multi_token_label_case_records_error_and_label_ids():
    case = next(c for c in SEED_CASES if c.label_style == "option-text")
    out = run_case(case, FakeRunner())
    assert out["error"]["type"] == "LabelTokenizationError"
    assert "not single tokens" in out["error"]["message"]
    assert "raw_probabilities" not in out and "answer" not in out
    assert any(len(ids) > 1 for ids in out["option_token_ids"].values())


def test_case_validation():
    with pytest.raises(ValueError, match="exactly one question"):
        FixtureCase("x", "", {"state": "s", "questions": {}})
    with pytest.raises(ValueError, match="needs a request prefix"):
        FixtureCase("x", "", {"state": "s", "questions": {"q": {}}}, prefix_cached=True)


def test_committed_fixtures_are_the_seed_set_on_gemma():
    fxs = load_fixtures(COMMITTED)
    assert [fx["id"] for fx in fxs] == sorted(c.id for c in SEED_CASES)
    for fx in fxs:
        assert fx["model"]["repo"] == GEMMA
        assert len(fx["model"]["revision"]) == 40 and len(fx["model"]["tokenizer_sha256"]) == 64
        assert fx["path"] == {"prefill": "split", "prefill_step": 2048}
    errors = [fx for fx in fxs if "error" in fx["expected"]]
    assert [fx["expected"]["error"]["type"] for fx in errors] == ["LabelTokenizationError"]


# --- on the real model ----------------------------------------------------------------------

needs_gemma = pytest.mark.skipif(not _HAVE_GEMMA, reason=f"{GEMMA} weights not in HF cache")


@pytest.fixture(scope="module")
def mlx_runner():
    pytest.importorskip("mlx.core")
    from mlx_classifiers.mlx_runner import MLXRunner

    return MLXRunner.load(GEMMA)


@pytest.mark.mlx
@needs_gemma
def test_generate_on_mlx_is_byte_identical(mlx_runner, tmp_path):
    """Determinism (4.1): two MLXRunner generations write identical bytes, and
    reproduce the committed fixtures' expected blocks."""
    a = generate(tmp_path / "a", mlx_runner)
    b = generate(tmp_path / "b", mlx_runner)
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_bytes() == pb.read_bytes()
    regenerated = {fx["id"]: fx for fx in load_fixtures(tmp_path / "a")}
    for fx in load_fixtures(COMMITTED):
        assert regenerated[fx["id"]]["expected"]["tokens"] == fx["expected"]["tokens"]
