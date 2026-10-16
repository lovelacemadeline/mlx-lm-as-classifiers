"""The static decision battery: labelled wire requests (Epic 5).

Story 5.1 owns the format and its loader. A battery is JSONL; each line is a
wire-schema :class:`~mlx_classifiers.schema.SystemOneRequest` plus three
reserved keys, stripped before the request is validated:

- ``id`` -- unique case id;
- ``gold`` -- per question name, the right answer: an option name for
  ``choice``, ``true``/``false`` for ``noul`` (true = yes), a level index for
  ``score``;
- ``tags`` -- string labels; ``source`` and ``difficulty`` are required.
  Perturbation groups add ``group`` (shared by every member, all with the same
  gold) and ``variant`` (``base`` for the unperturbed member).

``battery/seed.jsonl`` is the seed set, built by ``battery/build_seed.py``.

Story 5.2 owns the runner (``mlx-classifiers battery``): each case through
:func:`~mlx_classifiers.decide.decide`, then accuracy per question type and
tag, top-label ECE raw and calibrated (a ``--calibration`` config's
temperature, plus a temperature fitted on the battery itself with a held-out
split via :func:`~mlx_classifiers.fitting.calibrate_records`), position-bias
spread (each ``choice`` question re-asked with its options in the
:func:`~mlx_classifiers.decide.permutation_orders` orderings: the mean over
orderings of the largest per-option probability shift from the declared order)
and the flip rate of perturbation-group variants against their base. Accuracy
is the raw argmax, which temperature scaling does not change.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .calibrate import expected_calibration_error, softmax
from .decide import decide, permutation_orders
from .fitting import CalibrationConfig, calibrate_records
from .prompt import question_options
from .schema import ChoiceQuestion, NoulQuestion, SystemOneRequest, SystemOneResponse

__all__ = [
    "RESERVED_KEYS",
    "BatteryCase",
    "BatteryError",
    "BatteryReport",
    "QuestionResult",
    "add_arguments",
    "calibration_records",
    "gold_index",
    "load_battery",
    "main",
    "parse_case",
    "render_markdown",
    "run_battery",
    "run_case",
    "summarise",
    "write_report",
]

RESERVED_KEYS = ("id", "gold", "tags")
REQUIRED_TAGS = ("source", "difficulty")


class BatteryError(ValueError):
    """A battery line is malformed: bad JSON, an invalid request, or a bad gold."""


@dataclass(frozen=True)
class BatteryCase:
    """One battery line. ``wire`` is the request exactly as written (reserved keys
    removed); ``gold`` maps each question name to its gold option index in
    :func:`~mlx_classifiers.prompt.question_options` order."""

    id: str
    request: SystemOneRequest
    wire: dict[str, Any]
    gold: dict[str, int]
    tags: dict[str, str]

    @property
    def group(self) -> str | None:
        return self.tags.get("group")


def gold_index(question: Any, gold: Any) -> int:
    """The index of ``gold`` among ``question``'s options; :class:`BatteryError` if
    it is not one of them or is the wrong kind for the question type."""
    options = question_options(question)
    if isinstance(question, NoulQuestion):
        if not isinstance(gold, bool):
            raise BatteryError(f"noul gold must be true or false, got {gold!r}")
        return 0 if gold else 1
    if isinstance(question, ChoiceQuestion):
        if not isinstance(gold, str) or gold not in options:
            raise BatteryError(f"choice gold {gold!r} is not one of {list(options)}")
        return options.index(gold)
    if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(options):
        raise BatteryError(f"score gold {gold!r} is not a level index 0..{len(options) - 1}")
    return gold


def parse_case(obj: Mapping[str, Any]) -> BatteryCase:
    """Validate one decoded battery line."""
    if not isinstance(obj, Mapping):
        raise BatteryError("a battery line must be a JSON object")
    case_id, gold, tags = obj.get("id"), obj.get("gold"), obj.get("tags")
    if not isinstance(case_id, str) or not case_id:
        raise BatteryError("missing or empty 'id'")
    wire = {k: v for k, v in obj.items() if k not in RESERVED_KEYS}
    try:
        request = SystemOneRequest.model_validate(wire)
    except ValidationError as exc:
        raise BatteryError(f"{case_id}: not a valid request: {exc}") from None
    if not isinstance(gold, Mapping) or set(gold) != set(request.questions):
        raise BatteryError(
            f"{case_id}: gold must name exactly the questions {sorted(request.questions)}"
        )
    if not isinstance(tags, Mapping) or any(
        not isinstance(tags.get(t), str) for t in REQUIRED_TAGS
    ):
        raise BatteryError(f"{case_id}: tags must include string {list(REQUIRED_TAGS)}")
    if any(not isinstance(v, str) for v in tags.values()):
        raise BatteryError(f"{case_id}: tag values must be strings")
    try:
        golds = {name: gold_index(q, gold[name]) for name, q in request.questions.items()}
    except BatteryError as exc:
        raise BatteryError(f"{case_id}: {exc}") from None
    return BatteryCase(case_id, request, wire, golds, dict(tags))


def load_battery(path: str | Path) -> list[BatteryCase]:
    """Every case in a battery file. Raises :class:`BatteryError` naming
    ``file:line`` on the first bad line, a duplicate id, or a perturbation group
    whose members disagree on gold."""
    cases: list[BatteryCase] = []
    seen: set[str] = set()
    group_gold: dict[str, dict[str, int]] = {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            case = parse_case(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise BatteryError(f"{path}:{n}: not JSON ({exc.msg})") from None
        except BatteryError as exc:
            raise BatteryError(f"{path}:{n}: {exc}") from None
        if case.id in seen:
            raise BatteryError(f"{path}:{n}: duplicate id {case.id!r}")
        seen.add(case.id)
        if case.group is not None:
            want = group_gold.setdefault(case.group, case.gold)
            if want != case.gold:
                raise BatteryError(f"{path}:{n}: {case.id} disagrees with group {case.group}")
        cases.append(case)
    return cases


# --- the battery runner (story 5.2) -------------------------------------------

_FLOOR = 1e-12


def _raw_row(question: Any, answer: Any) -> list[float]:
    """The raw (``T = 1``) distribution over ``question_options(question)``."""
    if isinstance(question, NoulQuestion):
        p = float(answer.raw_noul if answer.raw_noul is not None else answer.noul)
        return [p, 1.0 - p]
    raw = answer.raw_probabilities or answer.probabilities
    return [float(raw[o]) for o in question_options(question)]


def _temper(raw: Sequence[float], temperature: float | None) -> list[float] | None:
    if temperature is None:
        return None
    return softmax([math.log(max(p, _FLOOR)) for p in raw], temperature)


def _argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=xs.__getitem__)


@dataclass
class QuestionResult:
    """One answered battery question. Probability rows are in option order."""

    case_id: str
    question: str
    type: str
    readout: str
    tags: dict[str, str]
    options: list[str]
    gold: int
    pred: int
    raw: list[float]
    calibrated: list[float] | None = None
    # choice questions run under --permutations N > 1:
    order_shift: float | None = None  # mean over orderings of max_o |p_k(o) - p_0(o)|
    order_flips: int | None = None  # orderings whose argmax differs from declared
    orderings: int | None = None

    @property
    def correct(self) -> bool:
        return self.pred == self.gold


def _reordered(case: BatteryCase, name: str, order: Sequence[int]) -> SystemOneRequest:
    """The case's request with only question ``name``, its options in ``order``."""
    wire = copy.deepcopy(case.wire)
    q = wire["questions"][name]
    items = list(q["criteria"].items())
    q["criteria"] = dict(items[i] for i in order)
    wire["questions"] = {name: q}
    return SystemOneRequest.model_validate(wire)


def run_case(
    case: BatteryCase,
    runner: Any,
    *,
    temperatures: Mapping[str, float] | None = None,
    permutations: int = 1,
) -> tuple[list[QuestionResult], list[str]]:
    """Answer one case; return its question results and any round-trip failures.

    The request is answered as written but with ``calibrated`` off, so the
    response carries raw probabilities; ``temperatures`` (per readout mode)
    gives the calibrated row. Round trip: the wire request re-serialises to the
    line, and the response re-validates to itself.
    """
    problems = []
    if case.request.to_wire() != case.wire:
        problems.append(f"{case.id}: request does not round-trip")
    readout = case.request.effective_options.readout
    temperature = (temperatures or {}).get(readout)
    resp = decide(case.request, runner)
    wire = resp.to_wire()
    if SystemOneResponse.model_validate(wire).to_wire() != wire:
        problems.append(f"{case.id}: response does not round-trip")
    results = []
    for name, question in case.request.questions.items():
        raw = _raw_row(question, resp.answers[name])
        r = QuestionResult(
            case_id=case.id,
            question=name,
            type=question.type,
            readout=readout,
            tags=case.tags,
            options=list(question_options(question)),
            gold=case.gold[name],
            pred=_argmax(raw),
            raw=raw,
            calibrated=_temper(raw, temperature),
        )
        if isinstance(question, ChoiceQuestion) and permutations > 1:
            names = list(question.criteria)
            shifts, flips = [], 0
            for order in permutation_orders(len(names), permutations, seed=name)[1:]:
                ans = decide(_reordered(case, name, order), runner).answers[name]
                p = [float((ans.raw_probabilities or ans.probabilities)[o]) for o in names]
                shifts.append(max(abs(a - b) for a, b in zip(p, raw, strict=True)))
                flips += _argmax(p) != r.pred
            r.order_shift = math.fsum(shifts) / len(shifts)
            r.order_flips, r.orderings = flips, len(shifts)
        results.append(r)
    return results, problems


def _accuracy(rs: Sequence[QuestionResult]) -> dict[str, Any]:
    return {"n": len(rs), "accuracy": sum(r.correct for r in rs) / len(rs) if rs else None}


def _ece(rs: Sequence[QuestionResult], attr: str, n_bins: int) -> float | None:
    rows = [getattr(r, attr) for r in rs]
    if not rs or any(row is None for row in rows):
        return None
    return expected_calibration_error(rows, [r.gold for r in rs], n_bins)


def calibration_records(results: Sequence[QuestionResult], model: str) -> list[dict[str, Any]]:
    """Results as labelled decision records, the input ``fitting`` reads."""
    return [
        {
            "model": model,
            "readout": r.readout,
            "options": r.options,
            "raw_probabilities": r.raw,
            "label": r.options[r.gold],
        }
        for r in results
    ]


def summarise(
    results: Sequence[QuestionResult], model: str, *, n_bins: int = 15, seed: int = 0
) -> dict[str, Any]:
    """Accuracy per type and tag, ECE raw / calibrated / battery-fitted, position
    bias and perturbation flip rate."""
    by_type: dict[str, list[QuestionResult]] = {}
    by_tag: dict[str, dict[str, list[QuestionResult]]] = {}
    for r in results:
        by_type.setdefault(r.type, []).append(r)
        for k, v in r.tags.items():
            if k in ("source", "difficulty", "variant"):
                by_tag.setdefault(k, {}).setdefault(v, []).append(r)

    types = {}
    for t, rs in sorted(by_type.items()):
        types[t] = {
            **_accuracy(rs),
            "ece_raw": _ece(rs, "raw", n_bins),
            "ece_calibrated": _ece(rs, "calibrated", n_bins),
            "mean_confidence": math.fsum(max(r.raw) for r in rs) / len(rs),
        }
    types["all"] = {
        **_accuracy(results),
        "ece_raw": _ece(results, "raw", n_bins),
        "ece_calibrated": _ece(results, "calibrated", n_bins),
        "mean_confidence": (
            math.fsum(max(r.raw) for r in results) / len(results) if results else None
        ),
    }

    fits = calibrate_records(calibration_records(results, model), seed=seed, n_bins=n_bins)
    fitted = [asdict(f) for f in fits]

    permuted = [r for r in results if r.orderings]
    position = None
    if permuted:
        position = {
            "n": len(permuted),
            "orderings": permuted[0].orderings + 1,
            "mean_shift": math.fsum(r.order_shift for r in permuted) / len(permuted),
            "max_shift": max(r.order_shift for r in permuted),
            "flip_rate": sum(r.order_flips for r in permuted) / sum(r.orderings for r in permuted),
        }

    groups: dict[str, list[QuestionResult]] = {}
    for r in results:
        if "group" in r.tags:
            groups.setdefault(f"{r.tags['group']}/{r.question}", []).append(r)
    flipped, n_variants, n_flips = [], 0, 0
    for key, rs in sorted(groups.items()):
        base = next((r for r in rs if r.tags.get("variant") == "base"), None)
        if base is None:
            continue
        variants = [r for r in rs if r is not base]
        flips = [r for r in variants if r.pred != base.pred]
        n_variants += len(variants)
        n_flips += len(flips)
        if flips:
            flipped.append(
                {
                    "group": key,
                    "gold": base.options[base.gold],
                    "base": [base.options[base.pred], round(max(base.raw), 3)],
                    "variants": {
                        r.tags.get("variant", r.case_id): [r.options[r.pred], round(max(r.raw), 3)]
                        for r in variants
                    },
                }
            )
    perturbation = None
    if groups:
        perturbation = {
            "groups": len(groups),
            "variants": n_variants,
            "variant_flip_rate": n_flips / n_variants if n_variants else None,
            "group_flip_rate": len(flipped) / len(groups),
            "flipped": flipped,
        }

    return {
        "n_questions": len(results),
        "by_type": types,
        "by_tag": {
            k: {v: _accuracy(rs) for v, rs in sorted(vs.items())}
            for k, vs in sorted(by_tag.items())
        },
        "battery_fit": fitted,
        "position_bias": position,
        "perturbation": perturbation,
    }


@dataclass
class BatteryReport:
    env: dict[str, Any]
    config: dict[str, Any]
    summary: dict[str, Any]
    results: list[QuestionResult]
    problems: list[str] = field(default_factory=list)


def run_battery(
    cases: Sequence[BatteryCase],
    runner: Any,
    *,
    temperatures: Mapping[str, float] | None = None,
    permutations: int = 1,
    env: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> BatteryReport:
    model = getattr(runner, "model_id", "default")
    results, problems = [], []
    for case in cases:
        rs, ps = run_case(case, runner, temperatures=temperatures, permutations=permutations)
        results += rs
        problems += ps
    return BatteryReport(
        env=env or {"model": model},
        config={
            "permutations": permutations,
            "temperatures": dict(temperatures or {}),
            **(config or {}),
        },
        summary=summarise(results, model),
        results=results,
        problems=problems,
    )


def _f(x: float | None, spec: str = ".3f") -> str:
    return "-" if x is None else format(x, spec)


def render_markdown(report: BatteryReport) -> str:
    env, cfg, s = report.env, report.config, report.summary
    lines = [
        "# Decision battery",
        "",
        f"- model: `{env.get('model')}`  git: `{env.get('git_rev', '-')}`",
        f"- battery: `{cfg.get('battery', '-')}` ({s['n_questions']} questions)",
        (
            f"- runner: {cfg.get('runner', '-')}  permutations: {cfg['permutations']}  "
            f"temperatures: {cfg['temperatures'] or 'none'}"
        ),
    ]
    if "vision_soft_tokens" in cfg:
        lines.append(f"- int8: {cfg.get('int8')}  vision_soft_tokens: {cfg['vision_soft_tokens']}")
    if "lowpowermode" in env:
        lines.append(f"- power: `{env['lowpowermode']}`; {env.get('power_source', '')}")
        lines.append(f"- machine: {env.get('machine')}  timestamp: {env.get('timestamp')}")
    lines += [
        f"- round-trip failures: {len(report.problems)}",
        "",
        "| type | n | accuracy | mean conf | ECE raw | ECE calibrated |",
        "|---|---|---|---|---|---|",
    ]
    for t, row in s["by_type"].items():
        lines.append(
            f"| {t} | {row['n']} | {_f(row['accuracy'])} | {_f(row['mean_confidence'])} "
            f"| {_f(row['ece_raw'])} | {_f(row['ece_calibrated'])} |"
        )
    if s["battery_fit"]:
        lines += [
            "",
            "Battery-fitted temperature (one per model and readout; ECE on the held-out split):",
            "",
            "| readout | train | held | T | ECE before | ECE after |",
            "|---|---|---|---|---|---|",
        ]
        for f in s["battery_fit"]:
            lines.append(
                f"| {f['readout']} | {f['n_train']} | {f['n_heldout']} | {f['temperature']:.3f} "
                f"| {f['ece_before']:.3f} | {f['ece_after']:.3f} |"
            )
    for tag, vals in s["by_tag"].items():
        lines += ["", f"| {tag} | n | accuracy |", "|---|---|---|"]
        lines += [f"| {v} | {row['n']} | {_f(row['accuracy'])} |" for v, row in vals.items()]
    pb = s["position_bias"]
    lines += ["", "## Position bias", ""]
    if pb:
        lines.append(
            f"{pb['n']} choice questions x {pb['orderings']} orderings: mean max-prob shift "
            f"{pb['mean_shift']:.3f} (max {pb['max_shift']:.3f}), answer flips in "
            f"{pb['flip_rate']:.1%} of permuted orderings."
        )
    else:
        lines.append("Not measured (run with `--permutations N`, N > 1).")
    pt = s["perturbation"]
    lines += ["", "## Prefix perturbation", ""]
    if pt:
        lines.append(
            f"{pt['groups']} groups, {pt['variants']} variants: "
            f"{_f(pt['variant_flip_rate'], '.1%')} of variants answer differently from their "
            f"base; {pt['group_flip_rate']:.1%} of groups have at least one flip."
        )
        if pt["flipped"]:
            lines += ["", "| group | gold | base (p) | variants (p) |", "|---|---|---|---|"]
            for g in pt["flipped"]:
                vs = ", ".join(f"{k}: {a} ({p})" for k, (a, p) in g["variants"].items())
                lines.append(
                    f"| {g['group']} | {g['gold']} | {g['base'][0]} ({g['base'][1]}) | {vs} |"
                )
    else:
        lines.append("No perturbation groups in this battery.")
    return "\n".join(lines) + "\n"


def write_report(report: BatteryReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md, js = out_dir / f"battery-{stamp}.md", out_dir / f"battery-{stamp}.json"
    md.write_text(render_markdown(report))
    body = {
        "env": report.env,
        "config": report.config,
        "summary": report.summary,
        "problems": report.problems,
        "results": [asdict(r) for r in report.results],
    }
    js.write_text(json.dumps(body, indent=2))
    return md, js


def add_arguments(p: argparse.ArgumentParser) -> None:
    from .mlx_runner import DEFAULT_MODEL

    p.add_argument("battery", type=Path, help="battery JSONL file (e.g. battery/seed.jsonl)")
    p.add_argument("--runner", choices=("fake", "mlx"), default="mlx")
    p.add_argument("--model", default=DEFAULT_MODEL, help="mlx-lm repo or path (--runner mlx)")
    p.add_argument(
        "--permutations",
        type=int,
        default=1,
        help="option orderings per choice question for the position-bias spread (1 = off)",
    )
    p.add_argument(
        "--calibration", type=Path, default=None, help="calibration config for calibrated ECE"
    )
    p.add_argument("--int8", action="store_true", help="W8A8 kernels (--runner mlx, M5)")
    p.add_argument(
        "--vision-tokens",
        type=int,
        choices=(70, 140, 280),
        default=None,
        help="soft-token budget per image (story 6.7; default: the checkpoint's 280)",
    )
    p.add_argument("--out", type=Path, default=Path("bench_runs"))
    p.set_defaults(func=main)


def _load_runner(
    kind: str, model: str, int8: bool = False, vision_tokens: int | None = None
) -> Any:
    if kind == "fake":
        from dataclasses import replace

        from .runner import FakeRunner

        runner = FakeRunner()
        if vision_tokens is not None:
            runner.image_tokens = replace(runner.image_tokens, max_soft_tokens=vision_tokens)
        return runner
    from .mlx_runner import MLXRunner

    return MLXRunner.load(model, int8=int8, vision_soft_tokens=vision_tokens)


def main(args: argparse.Namespace) -> int:
    from .bench import environment

    cases = load_battery(args.battery)
    int8, vision_tokens = getattr(args, "int8", False), getattr(args, "vision_tokens", None)
    runner = _load_runner(args.runner, args.model, int8, vision_tokens)
    model = getattr(runner, "model_id", args.model)
    temperatures: dict[str, float] = {}
    if args.calibration is not None:
        config = CalibrationConfig.load(args.calibration)
        for readout in ("first-token", "sequence"):
            t = config.temperature(model, readout)
            if t is not None:
                temperatures[readout] = t
        if not temperatures:
            print(f"warning: {args.calibration} has no temperature for {model}")
    report = run_battery(
        cases,
        runner,
        temperatures=temperatures,
        permutations=args.permutations,
        env=environment(model),
        config={
            "battery": str(args.battery),
            "runner": args.runner,
            "calibration": str(args.calibration) if args.calibration else None,
            "int8": int8,
            "vision_soft_tokens": vision_tokens or 280,
        },
    )
    md, js = write_report(report, args.out)
    print(md.read_text())
    print(f"wrote {md} and {js}")
    return 1 if report.problems else 0
