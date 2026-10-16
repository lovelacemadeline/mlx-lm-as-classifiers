"""Story 2.9 — the memory audit (``mlx-classifiers memory``) and, on the real models,
the step 2 gate: with ``int8=True`` the loaded model holds no stock copy of the
patched weights, so MLX active memory after the INT8 patch is within 10% of the
stock load (E2B, and the 26B when its weights are cached).
"""

from __future__ import annotations

import argparse
import sys

import pytest
from conftest import GEMMA, HF_HUB

from mlx_classifiers import memory

MOE = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"


def _cached(repo: str) -> bool:
    snaps = HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    return any(snaps.glob("*/*.safetensors"))


def _fake_audit() -> memory.Audit:
    readings = iter(
        [
            {"rss": 50, "footprint": 30, "max_rss": 50},
            {"rss": 3_850_000_000, "footprint": 3_800_000_000, "max_rss": 4_600_000_000},
            {"rss": 3_900_000_000, "footprint": 3_900_000_000, "max_rss": 4_600_000_000},
        ]
    )
    mlx = iter(
        [
            {"active": 0, "peak": 0, "cache": 0},
            {"active": 3_380_000_000, "peak": 3_380_000_000, "cache": 0},
            {"active": 3_500_000_000, "peak": 3_700_000_000, "cache": 100_000_000},
        ]
    )
    return memory.Audit(probe=lambda: next(readings), mlx=lambda: next(mlx))


def test_audit_renders_each_stage_and_its_change_from_the_previous_one():
    audit = _fake_audit()
    audit.mark("start")
    audit.mark("base weights", held=3_380_000_000)
    audit.mark("int8 patch", held=120_000_000, note="model arrays now 3.50 GB")
    lines = audit.render().splitlines()
    assert lines[0].startswith("| stage | footprint GB | dfootprint | rss GB | drss | active GB")
    assert len(lines) == 2 + 3
    base = lines[3].split(" | ")
    assert base[0] == "| base weights" and base[1:3] == ["3.80", "+3.80"]
    patch = lines[4].split(" | ")
    assert patch[1:3] == ["3.90", "+0.10"]  # footprint and its delta
    assert patch[5:7] == ["3.50", "+0.12"]  # MLX active and its delta
    assert patch[7:9] == ["0.10", "+0.10"]  # MLX cache
    assert patch[-2:] == ["0.12", "model arrays now 3.50 GB |"]  # held, note


@pytest.mark.skipif(sys.platform != "darwin", reason="mach task_info is macOS only")
def test_process_memory_reads_resident_and_footprint_on_macos():
    m = memory.process_memory()
    assert m["rss"] and m["footprint"] and m["max_rss"]
    assert 1 << 20 < m["rss"] <= m["max_rss"] * 1.5


def test_tree_nbytes_counts_underscored_arrays_once():
    mx = pytest.importorskip("mlx.core")
    from mlx import nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.zeros((4, 4), mx.float32)  # 64 bytes
            self._hidden = mx.zeros((8,), mx.int8)  # 8 bytes, skipped by parameters()
            self.again = [self.w]

    assert memory.tree_nbytes(M()) == 72


def test_cli_has_a_memory_command():
    p = argparse.ArgumentParser()
    memory.add_arguments(p)
    args = p.parse_args(["--int8", "--vision", "--delta-store-mb", "512"])
    assert (args.int8, args.vision, args.delta_store_mb, args.func) == (
        True,
        True,
        512,
        memory.main,
    )


@pytest.mark.mlx
@pytest.mark.parametrize("repo", [GEMMA, MOE], ids=["e2b", "26b"])
def test_int8_load_holds_no_stock_copy_and_stays_within_10pct(repo):
    pytest.importorskip("mlx.core")
    if not _cached(repo):
        pytest.skip(f"{repo} weights not in HF cache")
    from mlx_classifiers import moe

    if not moe.available():
        pytest.skip("needs MLX 0.32 on an M5 GPU")
    audit = memory.run_audit(repo, int8=True, entries=1, tokens=128)
    by = {s.stage: s for s in audit.samples}
    base, patched = by["base weights"], by["int8 patch"]
    print(f"[2.9] {repo}: active {base.active / 1e9:.2f} -> {patched.active / 1e9:.2f} GB")
    assert patched.active <= 1.10 * base.active
    assert patched.cache < 0.05 * base.active  # the freed stock buffers were released
