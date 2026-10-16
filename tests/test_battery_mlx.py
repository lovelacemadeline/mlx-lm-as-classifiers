"""Story 5.2 — the battery on the real Gemma 4 E2B (skipped when the HF cache lacks it)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import GEMMA, HF_HUB

mx = pytest.importorskip("mlx.core")

from mlx_classifiers.battery import load_battery, run_battery
from mlx_classifiers.mlx_runner import MLXRunner

SEED = Path(__file__).resolve().parents[1] / "battery" / "seed.jsonl"
_SNAP = HF_HUB / f"models--{GEMMA.replace('/', '--')}" / "snapshots"
pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        not any(_SNAP.glob("*/model.safetensors")), reason=f"{GEMMA} weights not in HF cache"
    ),
]


def test_battery_subset_on_mlx():
    """A small subset (every slice, two perturbation groups) runs end to end on the
    real model, round-trips, and gets the obvious probe row right."""
    cases = load_battery(SEED)
    keep = {"triage-01", "intent-01", "noul-01", "noul-02", "severity-19", "sentiment-25"}
    arc = [c for c in cases if c.id.startswith("arc-")][:4]
    subset = arc + [
        c for c in cases if c.id in keep or c.tags.get("group") in ("robot-03", "robot-05")
    ]
    report = run_battery(subset, MLXRunner.load(GEMMA), permutations=2)
    s = report.summary
    assert report.problems == []
    assert {"choice", "noul", "score"} <= set(s["by_type"])
    assert s["position_bias"]["n"] > 0 and s["perturbation"]["groups"] == 2
    fire = next(r for r in report.results if r.case_id == "robot-03")
    assert fire.correct and max(fire.raw) > 0.8
