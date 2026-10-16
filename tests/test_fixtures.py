"""Epic 4 — golden fixtures: format + generator (4.1) and checker (4.2)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from conftest import GEMMA, HF_HUB

from mlx_classifiers import FakeRunner
from mlx_classifiers.cli import main as cli_main
from mlx_classifiers.fixtures import (
    FIXTURE_FORMAT,
    RESULTS_FORMAT,
    SEED_CASES,
    FixtureCase,
    build_fixture,
    check,
    generate,
    load_fixtures,
    run_case,
)

COMMITTED = Path(__file__).resolve().parent.parent / "fixtures"
_SNAP = HF_HUB / f"models--{GEMMA.replace('/', '--')}" / "snapshots"
_HAVE_GEMMA = any(_SNAP.glob("*/model.safetensors"))


@pytest.fixture
def fake_dir(tmp_path):
    generate(tmp_path / "fx", FakeRunner())
    return tmp_path / "fx"


def _results(fixtures_dir: Path) -> dict:
    """A port's results file built from the fixtures themselves (a perfect port)."""
    cases = {fx["id"]: copy.deepcopy(fx["expected"]) for fx in load_fixtures(fixtures_dir)}
    for r in cases.values():
        r.get("error", {}).pop("message", None)  # ports need not match our wording
    return {"format": RESULTS_FORMAT, "cases": cases}


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
    with pytest.raises(ValueError, match="disagrees with label_style"):
        req = {"state": "s", "questions": {"q": {}}, "options": {"label_style": "option-text"}}
        FixtureCase("x", "", req)


def test_committed_fixtures_are_the_seed_set_on_gemma():
    fxs = load_fixtures(COMMITTED)
    assert [fx["id"] for fx in fxs] == sorted(c.id for c in SEED_CASES)
    for fx in fxs:
        assert fx["model"]["repo"] == GEMMA
        assert len(fx["model"]["revision"]) == 40 and len(fx["model"]["tokenizer_sha256"]) == 64
        assert fx["path"] == {"prefill": "split", "prefill_step": 2048}
    errors = [fx for fx in fxs if "error" in fx["expected"]]
    assert [fx["expected"]["error"]["type"] for fx in errors] == ["LabelTokenizationError"]


# --- 4.2: checker ---------------------------------------------------------------------------


def test_perfect_results_pass(fake_dir):
    report = check(fake_dir, _results(fake_dir))
    assert report.ok and len(report.cases) == len(SEED_CASES)
    assert all(c.token_ids_equal and c.max_prob_distance in (None, 0.0) for c in report.cases)


def test_in_process_callable_results(fake_dir):
    runner = FakeRunner()
    assert check(fake_dir, lambda fx: run_case(fx, runner)).ok


def test_one_wrong_token_id_names_position_and_both_ids(fake_dir):
    res = _results(fake_dir)
    case = res["cases"]["noul-first-token"]
    want = case["tokens"][7]
    case["tokens"][7] = want + 1
    report = check(fake_dir, res)
    assert not report.ok
    (bad,) = [c for c in report.cases if not c.ok]
    assert bad.id == "noul-first-token" and not bad.token_ids_equal
    assert bad.failures == [f"tokens[7]: expected token id {want}, got {want + 1}"]


def test_wrong_option_token_id_is_named(fake_dir):
    res = _results(fake_dir)
    ids = res["cases"]["choice-first-token-object-state"]["option_token_ids"]
    want = ids["technical"][0]
    ids["technical"] = [99]
    (bad,) = [c for c in check(fake_dir, res).cases if not c.ok]
    assert f"option_token_ids['technical'][0]: expected token id {want}, got 99" in bad.failures


def test_probability_tolerance(fake_dir):
    res = _results(fake_dir)
    probs = res["cases"]["score-first-token-no-prefix"]["raw_probabilities"]
    probs["0"] += 5e-4
    report = check(fake_dir, res)
    assert report.ok
    case = next(c for c in report.cases if c.id == "score-first-token-no-prefix")
    assert case.max_prob_distance == pytest.approx(5e-4)
    assert not check(fake_dir, res, prob_tol=1e-4).ok


def test_logits_reported_and_gated_only_on_request(fake_dir):
    res = _results(fake_dir)
    res["cases"]["noul-sequence"]["raw_logits"]["yes"] += 0.5
    assert check(fake_dir, res).ok
    report = check(fake_dir, res, logit_tol=0.1)
    (bad,) = [c for c in report.cases if not c.ok]
    assert bad.max_logit_distance == pytest.approx(0.5) and "raw_logits" in bad.failures[0]


def test_missing_case_wrong_answer_and_error_mismatch_fail(fake_dir):
    res = _results(fake_dir)
    del res["cases"]["noul-sequence"]
    res["cases"]["noul-first-token"]["answer"] = "maybe"
    err = res["cases"]["choice-first-token-multi-token-label-error"]
    err["error"]["type"] = "TooManyOptionsError"
    failed = {c.id: c.failures for c in check(fake_dir, res).cases if not c.ok}
    assert failed["noul-sequence"] == ["no result for this case"]
    assert any("answer: expected" in f for f in failed["noul-first-token"])
    assert failed["choice-first-token-multi-token-label-error"] == [
        "error.type: expected LabelTokenizationError, got TooManyOptionsError"
    ]


def test_port_that_answers_the_error_case_fails(fake_dir):
    res = _results(fake_dir)
    res["cases"]["choice-first-token-multi-token-label-error"].pop("error")
    failed = {c.id: c.failures for c in check(fake_dir, res).cases if not c.ok}
    assert failed == {"choice-first-token-multi-token-label-error": ["missing field 'error'"]}


def test_cli_check_exit_codes(fake_dir, tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_results(fake_dir)))
    assert cli_main(["fixtures", "check", str(fake_dir), "--results", str(good)]) == 0
    assert f"{len(SEED_CASES)}/{len(SEED_CASES)} cases passed" in capsys.readouterr().out
    res = _results(fake_dir)
    res["cases"]["noul-first-token"]["tokens"][3] = -1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(res))
    assert cli_main(["fixtures", "check", str(fake_dir), "--results", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "FAIL  noul-first-token" in out and "tokens[3]: expected token id" in out


def test_bare_mapping_results_and_format_guard(fake_dir):
    assert check(fake_dir, _results(fake_dir)["cases"]).ok
    with pytest.raises(ValueError, match="results format"):
        check(fake_dir, {"format": "other/9", "cases": {}})


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


@pytest.mark.mlx
@needs_gemma
def test_committed_fixtures_reproduce_on_mlx(mlx_runner):
    """The in-process checker mode (4.2): regenerate on MLXRunner and compare."""
    report = check(COMMITTED, lambda fx: run_case(fx, mlx_runner))
    assert report.ok, report.render()
