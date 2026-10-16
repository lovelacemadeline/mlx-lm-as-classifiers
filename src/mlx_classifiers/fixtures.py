"""Golden fixtures: the conformance contract for the Swift + vision port (ADR-0002, Epic 4).

A fixture pins, for one request and one question, everything a port must
reproduce: the compiled token ids, the readout position, the option -> token id
binding, the raw logits at the masked ids, the raw probabilities and the chosen
answer, at a fixed model revision and tokenizer hash, through MLXRunner's split
prefill. The format is documented in ``docs/reference/fixture-format.md``.

Generate (story 4.1)::

    mlx-classifiers fixtures generate fixtures/

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibrate import gather, softmax
from .errors import ClassifierError
from .prompt import compile_question
from .readout import bind_options, label_token_ids
from .schema import SystemOneRequest
from .sequence import score_sequences

__all__ = [
    "FIXTURE_FORMAT",
    "SEED_CASES",
    "FixtureCase",
    "build_fixture",
    "generate",
    "load_fixtures",
    "model_provenance",
    "run_case",
]

FIXTURE_FORMAT = "mlx-classifiers-fixture/1"

Result = dict[str, Any]


# --- seed cases ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureCase:
    """One seed case: a wire request with exactly one question.

    ``prefix_cached``: first-token mode KV-caches ``tokens[:prefix_len]`` (the
    system turn) and prefills the rest from it. ``label_style = "option-text"``
    renders the option texts as labels (the sequence-mode prompt) and then reads
    them out first-token, which must fail when a label is more than one token.
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


def run_case(case: FixtureCase | Mapping[str, Any], runner: Any) -> Result:
    """Run one case (a :class:`FixtureCase` or a loaded fixture dict) through
    ``runner`` and return the result block: the shape of a fixture's ``expected``.

    This is the reference a port mirrors: compile, bind in context, prefill
    (from the system-turn cache when ``prefix_cached``), gather, raw softmax.
    """
    request, prefix_cached, label_style = _case_fields(case)
    req = SystemOneRequest.model_validate(request)
    (name,) = req.questions
    readout = req.effective_options.readout
    render = "sequence" if label_style == "option-text" else readout
    prompt = compile_question(
        req, name, template=runner.chat_template, tokenize=runner.tokenize, readout=render
    )
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


# --- CLI ------------------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from .mlx_runner import DEFAULT_MODEL

    sub = parser.add_subparsers(dest="fixtures_command", required=True)
    g = sub.add_parser("generate", help="write the seed fixtures from MLXRunner (story 4.1)")
    g.add_argument("out_dir", type=Path)
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.set_defaults(func=_cmd_generate)


def _load_runner(repo: str) -> Any:
    from .mlx_runner import MLXRunner

    return MLXRunner.load(repo)


def _cmd_generate(args: argparse.Namespace) -> int:
    paths = generate(args.out_dir, _load_runner(args.model))
    print(f"wrote {len(paths)} fixtures to {args.out_dir}")
    return 0


def main(args: argparse.Namespace) -> int:
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    p = argparse.ArgumentParser(prog="mlx-classifiers fixtures")
    add_arguments(p)
    sys.exit(main(p.parse_args()))
