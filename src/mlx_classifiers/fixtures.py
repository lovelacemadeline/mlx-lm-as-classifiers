"""Golden fixtures: the conformance contract for the Swift + vision port (ADR-0002, Epic 4).

A fixture pins, for one request and one question, everything a port must
reproduce: the compiled token ids, the readout position, the option -> token id
binding, the raw logits at the masked ids, the raw probabilities and the chosen
answer, at a fixed model revision and tokenizer hash, through MLXRunner's split
prefill. The format is documented in ``docs/reference/fixture-format.md``.

Generate (story 4.1)::

    mlx-classifiers fixtures generate fixtures/

Check a port (story 4.2)::

    mlx-classifiers fixtures check fixtures/ --results port-results.json
    mlx-classifiers fixtures check fixtures/ --runner mlx      # regenerate + compare

or, from another repo, ``mlx_classifiers.fixtures.check(fixtures_dir, results)``.

Logits are pinned per *path*: naive ``model(all)[:, -1]`` and the split prefill
differ by up to 0.75 in raw logits while agreeing on argmax (story 2.5), so a
fixture records the split path and its cache boundary, and a port must follow it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .calibrate import gather, softmax
from .errors import ClassifierError
from .prompt import compile_question
from .prompt_cache import without_delta
from .readout import bind_options, label_token_ids
from .schema import SystemOneRequest
from .sequence import score_sequences

__all__ = [
    "FIXTURE_FORMAT",
    "RESULTS_FORMAT",
    "SEED_CASES",
    "CaseReport",
    "CheckReport",
    "FixtureCase",
    "build_fixture",
    "check",
    "compare",
    "generate",
    "load_fixtures",
    "model_provenance",
    "run_case",
]

FIXTURE_FORMAT = "mlx-classifiers-fixture/1"
RESULTS_FORMAT = "mlx-classifiers-results/1"
DEFAULT_PROB_TOL = 1e-3

Result = dict[str, Any]


# --- seed cases ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureCase:
    """One seed case: a wire request with exactly one question.

    ``prefix_cached``: first-token mode KV-caches ``tokens[:prefix_len]`` (the
    system turn) and prefills the rest from it. ``label_style = "option-text"``
    runs the case with the wire option ``options.label_style = "option-text"``
    (story 1.8): the option names are the labels, read out first-token, which must
    fail when a name is more than one token. A request that sets
    ``options.label_style`` itself must agree with ``label_style``.
    """

    id: str
    description: str
    request: dict[str, Any]
    prefix_cached: bool = False
    label_style: str = "letters"

    def __post_init__(self) -> None:
        if len(self.request.get("questions", {})) != 1:
            raise ValueError(f"case {self.id}: a fixture request holds exactly one question")
        if self.prefix_cached and not self.request.get("prefix"):
            raise ValueError(f"case {self.id}: prefix_cached needs a request prefix")
        if self.label_style not in ("letters", "option-text"):
            raise ValueError(f"case {self.id}: unknown label_style {self.label_style!r}")
        wire = self.request.get("options", {}).get("label_style", self.label_style)
        if wire != self.label_style:
            raise ValueError(
                f"case {self.id}: request options.label_style {wire!r} "
                f"disagrees with label_style {self.label_style!r}"
            )


_PREFIX = "You are the support triage router for a payments company. Answer precisely."
_TICKET = (
    "URGENT: none of our payouts have arrived for three days and our whole business is "
    "blocked. Every payment to our merchants is failing and we are losing money every hour."
)
_TICKET_OBJ = {
    "channel": "email",
    "customer": "Acme Ltd",
    "plan": "enterprise",
    "body": "Our invoice for March charged us twice. Please refund the duplicate charge.",
}
_TICKET_LIST = [
    "Customer: the app crashes every time I open the payouts tab.",
    "Agent: which version are you on?",
    "Customer: the latest one, and it started after the update yesterday.",
]
_NOUL = {"type": "noul", "instructions": "Is the ticket about payments?"}
_DEPT = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {
        "billing": "Payments, refunds, invoices, payouts",
        "technical": "Bugs, crashes, outages",
        "sales": "Pricing questions and new contracts",
    },
}
_SEVERITY = {
    "type": "score",
    "instructions": "How severe is the customer impact?",
    "criteria": ["trivial", "minor", "major", "critical"],
}
_DISPUTE = {
    "type": "choice",
    "instructions": "What kind of payment issue is this?",
    "criteria": {"refund": None, "chargeback": None, "fraud": None},
}


def _req(question: dict, state: Any = _TICKET, readout: str = "first-token", prefix=_PREFIX):
    req: dict[str, Any] = {"state": state, "questions": {"q": question}}
    if prefix is not None:
        req["prefix"] = prefix
    if readout != "first-token":
        req["options"] = {"readout": readout}
    return req


SEED_CASES: tuple[FixtureCase, ...] = (
    FixtureCase("noul-first-token", "Yes/No on a string state", _req(_NOUL)),
    FixtureCase(
        "choice-first-token-object-state",
        "3-way choice, state as a JSON object (sorted-key pretty JSON)",
        _req(_DEPT, _TICKET_OBJ),
    ),
    FixtureCase(
        "choice-first-token-cached-list-state",
        "3-way choice from a KV-cached system prefix, state as a list of blocks",
        _req(_DEPT, _TICKET_LIST),
        prefix_cached=True,
    ),
    FixtureCase(
        "score-first-token-no-prefix",
        "4-level score with no system turn (prefix_len 0)",
        _req(_SEVERITY, prefix=None),
    ),
    FixtureCase(
        "noul-sequence",
        "Yes/No scored by sequence loglikelihood",
        _req(_NOUL, "The office plants need watering on Fridays.", "sequence"),
    ),
    FixtureCase(
        "choice-sequence-multi-token-options",
        "choice over option texts, one of them two tokens, scored by sequence",
        _req(_DISPUTE, _TICKET_OBJ, "sequence"),
    ),
    FixtureCase(
        "score-sequence-object-state",
        "4-level score scored by sequence, state as a JSON object",
        _req(_SEVERITY, _TICKET_OBJ, "sequence"),
    ),
    FixtureCase(
        "choice-first-token-multi-token-label-error",
        "option texts as first-token labels; 'chargeback' is two tokens, so binding fails",
        _req(_DISPUTE, _TICKET_OBJ),
        label_style="option-text",
    ),
)


# --- running a case -------------------------------------------------------------------------


def _argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=lambda i: (xs[i], -i))  # first index wins ties


def _case_fields(case: FixtureCase | Mapping[str, Any]) -> tuple[dict, bool, str]:
    if isinstance(case, FixtureCase):
        return case.request, case.prefix_cached, case.label_style
    return case["request"], bool(case.get("prefix_cached")), case.get("label_style", "letters")


def _with_label_style(req: SystemOneRequest, label_style: str) -> SystemOneRequest:
    """``req`` with ``options.label_style`` set, so the case compiles through the
    same path as a wire request carrying that option."""
    opts = req.effective_options
    if opts.label_style == label_style:
        return req
    return req.model_copy(update={"options": opts.model_copy(update={"label_style": label_style})})


def run_case(case: FixtureCase | Mapping[str, Any], runner: Any) -> Result:
    """Run one case (a :class:`FixtureCase` or a loaded fixture dict) through
    ``runner`` and return the result block: the shape of a fixture's ``expected``.

    This is the reference a port mirrors: compile, bind in context, prefill
    (from the system-turn cache when ``prefix_cached``), gather, raw softmax.
    Delta prefix caching (story 3.4) is off: a case pins its cache path, and a
    hit from an earlier case would change the chunking (bf16 noise).
    """
    runner = without_delta(runner)
    request, prefix_cached, label_style = _case_fields(case)
    req = _with_label_style(SystemOneRequest.model_validate(request), label_style)
    (name,) = req.questions
    readout = req.effective_options.readout
    prompt = compile_question(req, name, template=runner.chat_template, tokenize=runner.tokenize)
    out: Result = {
        "tokens": list(prompt.tokens),
        "prefix_len": prompt.prefix_len,
        "readout_position": prompt.readout_position,
        "options": list(prompt.options),
        "labels": [prompt.label_for(i) for i in range(len(prompt.options))],
    }
    if readout == "sequence":
        s = score_sequences(prompt, runner)
        out["option_token_ids"] = s.option_token_ids
        logits = list(s.scores)
    else:
        try:
            binding = bind_options(prompt, runner.tokenize)
        except ClassifierError as e:
            ids = {}
            for o, lab in zip(prompt.options, out["labels"], strict=True):
                try:
                    ids[o] = list(label_token_ids(prompt.text, lab, runner.tokenize, prompt.tokens))
                except ClassifierError:
                    ids[o] = []
            out["option_token_ids"] = ids
            out["error"] = {"type": type(e).__name__, "message": str(e)}
            return out
        cache = runner.cache_prefix(prompt.prefix_tokens) if prefix_cached else None
        r = runner.prefill(prompt.tokens, cache, [prompt.readout_position])
        out["option_token_ids"] = binding.option_token_ids
        logits = gather(r.at(prompt.readout_position), binding.token_ids)
    probs = softmax(logits)
    out["raw_logits"] = dict(zip(prompt.options, logits, strict=True))
    out["raw_probabilities"] = dict(zip(prompt.options, probs, strict=True))
    out["answer"] = prompt.options[_argmax(probs)]
    return out


# --- generation (story 4.1) -----------------------------------------------------------------


def model_provenance(repo: str) -> dict[str, Any]:
    """``{repo, revision, tokenizer_sha256}`` from the local HF cache: the snapshot
    hash holding ``tokenizer.json`` and that file's sha256. ``None`` when unknown."""
    out: dict[str, Any] = {"repo": repo, "revision": None, "tokenizer_sha256": None}
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo, "tokenizer.json")
    except (ImportError, ValueError):  # not installed, or not a valid hub repo id
        return out
    if isinstance(path, str):
        p = Path(path)
        out["revision"] = p.parent.name
        out["tokenizer_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for pkg in ("mlx-classifiers", "mlx", "mlx-lm"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def build_fixture(
    case: FixtureCase, runner: Any, provenance: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The full fixture dict for ``case``: provenance, path, request, expected."""
    req = SystemOneRequest.model_validate(case.request)
    model_id = getattr(runner, "model_id", "unknown")
    return {
        "format": FIXTURE_FORMAT,
        "id": case.id,
        "description": case.description,
        "model": dict(provenance) if provenance is not None else model_provenance(model_id),
        "path": {"prefill": "split", "prefill_step": getattr(runner, "prefill_step", None)},
        "readout": req.effective_options.readout,
        "label_style": case.label_style,
        "prefix_cached": case.prefix_cached,
        "question": next(iter(req.questions)),
        "request": case.request,
        "expected": run_case(case, runner),
        "generator": _versions(),
    }


# Only arrays json.dumps broke over lines: a JSON string never holds a raw newline.
_INT_ARRAY = re.compile(r"\[\n\s*(-?\d+(?:,\n\s*-?\d+)*)\n\s*\]")


def _dump(obj: Any) -> str:
    """Indented JSON with integer arrays (token ids) kept on one line."""
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    one_line = lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]"
    return _INT_ARRAY.sub(one_line, text) + "\n"


def generate(
    out_dir: str | Path,
    runner: Any,
    cases: Iterable[FixtureCase] = SEED_CASES,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write one ``<case id>.json`` per case into ``out_dir``; return the paths.

    Output is a pure function of the runner's outputs (no timestamps, fixed key
    order), so two runs on the same machine are byte-identical.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if provenance is None:
        provenance = model_provenance(getattr(runner, "model_id", "unknown"))
    paths = []
    for case in cases:
        path = out / f"{case.id}.json"
        path.write_text(_dump(build_fixture(case, runner, provenance)), encoding="utf-8")
        paths.append(path)
    return paths


def load_fixtures(directory: str | Path) -> list[dict[str, Any]]:
    """Every ``*.json`` fixture in ``directory``, sorted by file name."""
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no fixture files in {directory}")
    out = []
    for p in paths:
        fx = json.loads(p.read_text(encoding="utf-8"))
        if fx.get("format") != FIXTURE_FORMAT:
            raise ValueError(f"{p}: format {fx.get('format')!r}, expected {FIXTURE_FORMAT!r}")
        out.append(fx)
    return out


# --- checking (story 4.2) -------------------------------------------------------------------


@dataclass
class CaseReport:
    """One case's comparison. ``token_ids_equal`` covers the prompt tokens, the
    readout position and the option token ids (all exact)."""

    id: str
    failures: list[str] = field(default_factory=list)
    token_ids_equal: bool = True
    max_prob_distance: float | None = None
    max_logit_distance: float | None = None

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        dp = "-" if self.max_prob_distance is None else f"{self.max_prob_distance:.3g}"
        dl = "-" if self.max_logit_distance is None else f"{self.max_logit_distance:.3g}"
        tok = "exact" if self.token_ids_equal else "MISMATCH"
        head = f"{'PASS' if self.ok else 'FAIL'}  {self.id}  token ids {tok}  max|dp| {dp}"
        lines = [f"{head}  max|dlogit| {dl}"]
        lines += [f"      - {f}" for f in self.failures]
        return "\n".join(lines)


@dataclass
class CheckReport:
    cases: list[CaseReport]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.cases)

    def render(self) -> str:
        passed = sum(c.ok for c in self.cases)
        body = [c.render() for c in self.cases]
        return "\n".join([*body, f"{passed}/{len(self.cases)} cases passed"])


def _compare_ids(where: str, want: Sequence[int], got: Any, failures: list[str]) -> bool:
    if not isinstance(got, list):
        failures.append(f"{where}: expected a list of {len(want)} token ids, got {got!r}")
        return False
    ok = True
    for i, (w, g) in enumerate(zip(want, got, strict=False)):
        if w != g:
            failures.append(f"{where}[{i}]: expected token id {w}, got {g}")
            ok = False
            break  # the first divergence; later ones usually follow from it
    if len(want) != len(got):
        failures.append(f"{where}: expected {len(want)} token ids, got {len(got)}")
        ok = False
    return ok


def _max_distance(
    where: str, want: Mapping[str, float], got: Any, failures: list[str]
) -> tuple[float | None, str | None]:
    if not isinstance(got, Mapping) or set(got) != set(want):
        keys = sorted(got) if isinstance(got, Mapping) else got
        failures.append(f"{where}: expected options {sorted(want)}, got {keys}")
        return None, None
    worst = max(want, key=lambda o: abs(float(got[o]) - want[o]))
    return abs(float(got[worst]) - want[worst]), worst


def compare(
    expected: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    *,
    case_id: str = "?",
    prob_tol: float = DEFAULT_PROB_TOL,
    logit_tol: float | None = None,
) -> CaseReport:
    """Compare one result block against a fixture's ``expected`` block.

    Exact: ``tokens``, ``prefix_len``, ``readout_position``, ``options``,
    ``labels``, ``option_token_ids``, ``answer``, ``error.type``. Within
    ``prob_tol`` (max absolute difference): ``raw_probabilities``. ``raw_logits``
    distance is reported, and fails only when ``logit_tol`` is given.
    """
    rep = CaseReport(case_id)
    f = rep.failures
    if result is None:
        f.append("no result for this case")
        rep.token_ids_equal = False
        return rep
    for key in expected:
        if key not in result:
            f.append(f"missing field {key!r}")
    if "error" in result and "error" not in expected:
        f.append(f"unexpected error {result['error']!r}")

    tok_ok = True
    if "tokens" in result:
        tok_ok &= _compare_ids("tokens", expected["tokens"], result["tokens"], f)
    for key in ("prefix_len", "readout_position"):
        if key in result and result[key] != expected[key]:
            f.append(f"{key}: expected {expected[key]}, got {result[key]}")
            tok_ok &= key != "readout_position"
    for key in ("options", "labels"):
        if key in result and list(result[key]) != list(expected[key]):
            f.append(f"{key}: expected {expected[key]}, got {result[key]}")
    if isinstance(result.get("option_token_ids"), Mapping):
        got_ids = result["option_token_ids"]
        for o, want in expected["option_token_ids"].items():
            if o not in got_ids:
                f.append(f"option_token_ids: no ids for option {o!r}")
                tok_ok = False
            else:
                tok_ok &= _compare_ids(f"option_token_ids[{o!r}]", want, got_ids[o], f)
    elif "option_token_ids" in result:
        f.append(f"option_token_ids: expected an object, got {result['option_token_ids']!r}")
        tok_ok = False
    rep.token_ids_equal = tok_ok and "tokens" in result and "option_token_ids" in result

    if "error" in expected:
        got = result.get("error")
        want_type = expected["error"]["type"]
        got_type = got.get("type") if isinstance(got, Mapping) else None
        if got is not None and got_type != want_type:
            f.append(f"error.type: expected {want_type}, got {got_type}")
        return rep

    if "raw_probabilities" in result:
        want, got = expected["raw_probabilities"], result["raw_probabilities"]
        d, o = _max_distance("raw_probabilities", want, got, f)
        rep.max_prob_distance = d
        if d is not None and not d <= prob_tol:
            f.append(f"raw_probabilities: max distance {d:.3g} at {o!r} exceeds {prob_tol:g}")
    if "raw_logits" in result:
        d, o = _max_distance("raw_logits", expected["raw_logits"], result["raw_logits"], f)
        rep.max_logit_distance = d
        if d is not None and logit_tol is not None and not d <= logit_tol:
            f.append(f"raw_logits: max distance {d:.3g} at {o!r} exceeds {logit_tol:g}")
    if "answer" in result and result["answer"] != expected["answer"]:
        f.append(f"answer: expected {expected['answer']!r}, got {result['answer']!r}")
    return rep


ResultsSource = str | Path | Mapping[str, Any] | Callable[[dict[str, Any]], Result]


def _results_lookup(results: ResultsSource) -> Callable[[dict[str, Any]], Result | None]:
    if callable(results):
        return results
    if isinstance(results, str | Path):
        results = json.loads(Path(results).read_text(encoding="utf-8"))
    if "format" in results:
        if results["format"] != RESULTS_FORMAT:
            raise ValueError(f"results format {results['format']!r}, expected {RESULTS_FORMAT!r}")
        results = results["cases"]
    return lambda fx: results.get(fx["id"])


def check(
    fixtures: str | Path | Sequence[Mapping[str, Any]],
    results: ResultsSource,
    *,
    prob_tol: float = DEFAULT_PROB_TOL,
    logit_tol: float | None = None,
) -> CheckReport:
    """Check a port's results against fixtures.

    ``fixtures`` is a directory or a list of loaded fixture dicts. ``results`` is
    a results JSON path, its parsed object (``{"format", "cases": {id: result}}``
    or a bare ``{id: result}``), or a callable ``fixture -> result`` (run the
    port in process). A result has the shape of the fixture's ``expected`` block.
    """
    fxs = load_fixtures(fixtures) if isinstance(fixtures, str | Path) else list(fixtures)
    lookup = _results_lookup(results)
    reports = []
    for fx in fxs:
        reports.append(
            compare(
                fx["expected"],
                lookup(fx),
                case_id=fx["id"],
                prob_tol=prob_tol,
                logit_tol=logit_tol,
            )
        )
    return CheckReport(reports)


# --- CLI ------------------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from .mlx_runner import DEFAULT_MODEL

    sub = parser.add_subparsers(dest="fixtures_command", required=True)
    g = sub.add_parser("generate", help="write the seed fixtures from MLXRunner (story 4.1)")
    g.add_argument("out_dir", type=Path)
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.set_defaults(func=_cmd_generate)
    c = sub.add_parser("check", help="compare a port's results with the fixtures (story 4.2)")
    c.add_argument("fixtures_dir", type=Path)
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", type=Path, help="results JSON written by the port")
    src.add_argument("--runner", choices=["mlx"], help="regenerate in process and compare")
    c.add_argument("--model", default=None, help="with --runner mlx (default: the fixtures')")
    c.add_argument("--prob-tol", type=float, default=DEFAULT_PROB_TOL)
    c.add_argument("--logit-tol", type=float, default=None, help="also gate raw logits")
    c.add_argument(
        "--int8", action="store_true", help="with --runner mlx: load the W8A8 kernels (2.3)"
    )
    c.set_defaults(func=_cmd_check)


def _load_runner(repo: str, int8: bool = False) -> Any:
    from .mlx_runner import MLXRunner

    return MLXRunner.load(repo, int8=int8)


def _cmd_generate(args: argparse.Namespace) -> int:
    paths = generate(args.out_dir, _load_runner(args.model))
    print(f"wrote {len(paths)} fixtures to {args.out_dir}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    fxs = load_fixtures(args.fixtures_dir)
    if args.runner == "mlx":
        runner = _load_runner(args.model or fxs[0]["model"]["repo"], int8=args.int8)
        results: ResultsSource = lambda fx: run_case(fx, runner)
    else:
        results = args.results
    report = check(fxs, results, prob_tol=args.prob_tol, logit_tol=args.logit_tol)
    print(report.render())
    return 0 if report.ok else 1


def main(args: argparse.Namespace) -> int:
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    p = argparse.ArgumentParser(prog="mlx-classifiers fixtures")
    add_arguments(p)
    sys.exit(main(p.parse_args()))
