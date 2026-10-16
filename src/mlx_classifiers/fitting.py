"""Fit calibration temperatures from labelled decisions (story 3.2).

Input is JSONL, one decision (one answered question) per line, in the decision
log format (story 3.3) with a ``label`` added. The fields read here are::

    {"model": "...", "readout": "first-token", "options": ["billing", "technical"],
     "raw_probabilities": [0.1, 0.9], "label": "technical"}

``raw_probabilities`` are the uncalibrated (``T = 1``) softmax over the
declared options, so ``log(raw)`` is the option-logit row up to a constant,
which softmax ignores: fitting ``T`` on it is fitting ``T`` on the logits. (With
``options.permutations > 1`` the logged row is the mean over orderings, and
``log(mean)`` is the natural row to calibrate, since that mean is what is
served.) Lines without a ``label`` are skipped.

One temperature is fitted per ``(model, readout)`` on a seeded train split,
and ECE / NLL before (``T = 1``) and after are reported on the held-out split.
Results persist to a :class:`CalibrationConfig` JSON file the server loads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .calibrate import TemperatureScaler, expected_calibration_error, nll, softmax

__all__ = [
    "CALIBRATION_ENV",
    "CalibrationConfig",
    "GroupFit",
    "add_arguments",
    "calibrate_records",
    "default_calibration_path",
    "label_index",
    "main",
    "read_jsonl",
]

CALIBRATION_ENV = "MLX_CLASSIFIERS_CALIBRATION"
_FLOOR = 1e-12  # log(0) guard for a probability that underflowed


def default_calibration_path() -> Path:
    """``$MLX_CLASSIFIERS_CALIBRATION``, else ``calibration.json`` in the cwd."""
    return Path(os.environ.get(CALIBRATION_ENV, "calibration.json"))


class CalibrationConfig:
    """Fitted temperatures per model and readout mode, persisted as JSON::

        {"version": 1,
         "models": {"<model id>": {"first-token": {"temperature": 1.8, "n_train": 700,
                                                   "n_heldout": 300, "ece_before": ...,
                                                   "ece_after": ..., "fitted_at": "..."}}}}

    Only ``temperature`` is read back; the rest is the audit trail of the fit.
    """

    VERSION = 1

    def __init__(self, models: dict[str, dict[str, dict[str, Any]]] | None = None):
        self.models: dict[str, dict[str, dict[str, Any]]] = models or {}

    @classmethod
    def load(cls, path: str | Path) -> CalibrationConfig:
        data = json.loads(Path(path).read_text())
        if data.get("version") != cls.VERSION:
            raise ValueError(
                f"{path}: unsupported calibration config version {data.get('version')}"
            )
        return cls(data.get("models", {}))

    @classmethod
    def load_or_empty(cls, path: str | Path) -> CalibrationConfig:
        return cls.load(path) if Path(path).exists() else cls()

    def save(self, path: str | Path) -> None:
        body = {"version": self.VERSION, "models": self.models}
        Path(path).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")

    def temperature(self, model: str, readout: str) -> float | None:
        entry = self.models.get(model, {}).get(readout)
        return float(entry["temperature"]) if entry else None

    def set(self, model: str, readout: str, entry: dict[str, Any]) -> None:
        self.models.setdefault(model, {})[readout] = entry


@dataclass(frozen=True)
class GroupFit:
    """One ``(model, readout)`` fit; ECE/NLL are measured on the held-out split."""

    model: str
    readout: str
    temperature: float
    n_train: int
    n_heldout: int
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float


def read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{n}: not JSON ({exc.msg})") from None
    return out


def label_index(record: dict[str, Any]) -> int:
    """The index of ``record["label"]`` in ``record["options"]``.

    A label is an option name; for convenience a bool labels a ``noul``
    (``true`` = ``"yes"``) and an int labels a ``score`` level by index.
    """
    options, label = list(record["options"]), record["label"]
    if isinstance(label, bool):
        label = "yes" if label else "no"
    elif isinstance(label, int):
        label = str(label)
    if label not in options:
        raise ValueError(f"label {record['label']!r} is not one of {options}")
    return options.index(label)


def _row(record: dict[str, Any]) -> list[float]:
    raw = record["raw_probabilities"]
    if len(raw) != len(record["options"]):
        raise ValueError("raw_probabilities and options differ in length")
    return [math.log(max(float(p), _FLOOR)) for p in raw]


def calibrate_records(
    records: Sequence[dict[str, Any]],
    *,
    heldout: float = 0.3,
    seed: int = 0,
    n_bins: int = 15,
) -> list[GroupFit]:
    """Fit one temperature per ``(model, readout)`` over the labelled ``records``.

    Each group is shuffled with ``seed`` and split; the held-out share is
    ``heldout`` (at least one record, leaving at least one to fit). Groups with
    fewer than two labelled records are skipped.
    """
    if not 0 < heldout < 1:
        raise ValueError("heldout must be in (0, 1)")
    groups: dict[tuple[str, str], list[tuple[list[float], int]]] = {}
    for rec in records:
        if rec.get("label") is None:
            continue
        key = (str(rec["model"]), str(rec.get("readout", "first-token")))
        groups.setdefault(key, []).append((_row(rec), label_index(rec)))

    fits = []
    for (model, readout), rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        rows = list(rows)
        random.Random(seed).shuffle(rows)
        n_held = min(max(1, round(len(rows) * heldout)), len(rows) - 1)
        held, train = rows[:n_held], rows[n_held:]
        scaler = TemperatureScaler()
        scaler.fit([r for r, _ in train], [y for _, y in train], n_bins=n_bins)
        t = scaler.temperature
        h_rows, h_labels = [r for r, _ in held], [y for _, y in held]
        fits.append(
            GroupFit(
                model=model,
                readout=readout,
                temperature=t,
                n_train=len(train),
                n_heldout=len(held),
                ece_before=expected_calibration_error(
                    [softmax(r) for r in h_rows], h_labels, n_bins
                ),
                ece_after=expected_calibration_error(
                    [softmax(r, t) for r in h_rows], h_labels, n_bins
                ),
                nll_before=nll(h_rows, h_labels, 1.0),
                nll_after=nll(h_rows, h_labels, t),
            )
        )
    return fits


def render_table(fits: Sequence[GroupFit]) -> str:
    head = f"{'model':<40} {'readout':<12} {'train':>6} {'held':>6} {'T':>7} "
    head += f"{'ECE before':>11} {'ECE after':>10} {'NLL before':>11} {'NLL after':>10}"
    lines = [head, "-" * len(head)]
    for f in fits:
        lines.append(
            f"{f.model:<40} {f.readout:<12} {f.n_train:>6} {f.n_heldout:>6} "
            f"{f.temperature:>7.3f} {f.ece_before:>11.4f} {f.ece_after:>10.4f} "
            f"{f.nll_before:>11.4f} {f.nll_after:>10.4f}"
        )
    return "\n".join(lines)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("decisions", nargs="+", type=Path, help="labelled decisions JSONL file(s)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"calibration config to update (default: ${CALIBRATION_ENV} or ./calibration.json)",
    )
    p.add_argument("--heldout", type=float, default=0.3, help="held-out share for ECE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=15, help="ECE confidence bins")


def main(args: argparse.Namespace) -> int:
    records = read_jsonl(args.decisions)
    labelled = sum(1 for r in records if r.get("label") is not None)
    fits = calibrate_records(records, heldout=args.heldout, seed=args.seed, n_bins=args.bins)
    print(f"{len(records)} decisions, {labelled} labelled")
    if not fits:
        print("nothing to fit: need at least two labelled decisions per (model, readout)")
        return 1
    print("ECE / NLL measured on the held-out split")
    print(render_table(fits))
    out = args.out or default_calibration_path()
    config = CalibrationConfig.load_or_empty(out)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for f in fits:
        entry = {k: v for k, v in asdict(f).items() if k not in ("model", "readout")}
        config.set(f.model, f.readout, {**entry, "fitted_at": now})
    config.save(out)
    print(f"wrote {out}")
    return 0
