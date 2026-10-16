"""Masked softmax and single-scalar temperature calibration.

Pure Python on purpose: the masked row is a handful of floats, and the Swift
port reimplements these few lines verbatim. Everything is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "CalibrationReport",
    "TemperatureScaler",
    "expected_calibration_error",
    "gather",
    "nll",
    "softmax",
]


def gather(row, token_ids: Sequence[int]) -> list[float]:
    """The logits of ``token_ids`` from a vocab row (the option mask)."""
    return [float(row[t]) for t in token_ids]


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax of ``logits / temperature``."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = math.fsum(exps)
    return [e / total for e in exps]


def _log_softmax_at(logits: Sequence[float], index: int, temperature: float) -> float:
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    return scaled[index] - m - math.log(math.fsum(math.exp(x - m) for x in scaled))


def nll(logit_rows: Sequence[Sequence[float]], labels: Sequence[int], temperature: float = 1.0):
    """Mean negative log-likelihood of ``labels`` under ``softmax(rows / T)``."""
    if len(logit_rows) != len(labels) or not labels:
        raise ValueError("need one label per logit row, and at least one row")
    total = math.fsum(
        -_log_softmax_at(row, y, temperature) for row, y in zip(logit_rows, labels, strict=True)
    )
    return total / len(labels)


def expected_calibration_error(
    prob_rows: Sequence[Sequence[float]], labels: Sequence[int], n_bins: int = 15
) -> float:
    """Top-label ECE with ``n_bins`` equal-width confidence bins.

    Each prediction's confidence is its max probability; ECE is the
    sample-weighted mean of ``|accuracy - mean confidence|`` over the bins.
    """
    if len(prob_rows) != len(labels) or not labels:
        raise ValueError("need one label per probability row, and at least one row")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for probs, y in zip(prob_rows, labels, strict=True):
        conf = max(probs)
        pred = probs.index(conf)
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, pred == y))
    ece = 0.0
    for members in bins:
        if members:
            acc = sum(ok for _, ok in members) / len(members)
            conf = math.fsum(c for c, _ in members) / len(members)
            ece += len(members) / len(labels) * abs(acc - conf)
    return ece


@dataclass(frozen=True)
class CalibrationReport:
    temperature: float
    n: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float


class TemperatureScaler:
    """Fit one temperature ``T`` by minimising NLL on labelled decisions.

    NLL is convex in ``1/T``, so a golden-section search over ``log T`` finds the
    global optimum deterministically. ``T > 1`` softens an over-confident model.
    """

    def __init__(self, temperature: float = 1.0, bounds: tuple[float, float] = (0.05, 20.0)):
        self.temperature = temperature
        self.bounds = bounds
        self.report: CalibrationReport | None = None

    def fit(
        self,
        logit_rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        n_bins: int = 15,
        iterations: int = 100,
    ) -> CalibrationReport:
        lo, hi = (math.log(b) for b in self.bounds)
        g = (math.sqrt(5) - 1) / 2

        def f(u: float) -> float:
            return nll(logit_rows, labels, math.exp(u))

        a, b = lo, hi
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(iterations):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = f(d)
        self.temperature = math.exp((a + b) / 2)
        before = [softmax(r) for r in logit_rows]
        after = [softmax(r, self.temperature) for r in logit_rows]
        self.report = CalibrationReport(
            temperature=self.temperature,
            n=len(labels),
            nll_before=nll(logit_rows, labels, 1.0),
            nll_after=nll(logit_rows, labels, self.temperature),
            ece_before=expected_calibration_error(before, labels, n_bins),
            ece_after=expected_calibration_error(after, labels, n_bins),
        )
        return self.report

    def apply(self, logits: Sequence[float]) -> list[float]:
        return softmax(logits, self.temperature)
