"""Story 6.2 — the vendored Gemma 4 vision path: provenance, no mlx-vlm, loads and runs."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "mlx_classifiers"
VISION = PKG / "vision"
VENDORED = ("encoder.py", "projector.py", "processor.py", "merge.py")


@pytest.mark.parametrize("name", VENDORED)
def test_vendored_files_carry_provenance_headers(name):
    head = (VISION / name).read_text().split('"""', 1)[0]
    assert "mlx-vlm 0.7.8" in head and "MIT" in head
    assert "sha256 " in head and "LICENSE.mlx-vlm" in head


def test_mlx_vlm_licence_text_ships_beside_the_code():
    text = (VISION / "LICENSE.mlx-vlm").read_text()
    assert text.startswith("MIT License") and "Prince Canuma" in text


def test_no_module_imports_mlx_vlm():
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n.split(".")[0] == "mlx_vlm" for n in names), path


_TINY = r"""
import sys
sys.modules["mlx_vlm"] = None  # any import of it now fails
import mlx.core as mx
from mlx_classifiers.images import write_png
from mlx_classifiers.vision.tower import VisionTower

config = {"vision_soft_tokens_per_image": 4, "vision_config": {
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
    "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
    "position_embedding_size": 64, "default_output_length": 4, "use_clipped_linears": True}}
mx.random.seed(0)
tower = VisionTower(config, text_hidden_size=24)
png = write_png(40, 40, [[(255, 0, 0)] * 40 for _ in range(40)])
a = tower.encode_png(png)
b = tower.encode_pixels(tower.pixel_values(png))[0]
assert a.shape == (4, 24), a.shape
assert mx.array_equal(a, b).item()
assert tower.encodes == 1 and tower.encode_png(png) is a
assert "mlx_vlm" not in [m for m in sys.modules if sys.modules[m] is not None]
print("ok", tuple(tower.pixel_values(png).shape))
"""


def test_vision_path_runs_with_mlx_vlm_unimportable():
    pytest.importorskip("mlx.core")
    pytest.importorskip("PIL")
    out = subprocess.run(
        [sys.executable, "-c", _TINY], check=False, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok (1, 3, 96, 96)"


def test_text_only_runner_import_loads_nothing_vision_related():
    pytest.importorskip("mlx_lm")
    code = (
        "import sys; import mlx_classifiers.mlx_runner, mlx_classifiers.decide, "
        "mlx_classifiers.server; "
        "print(sorted(m for m in sys.modules if m.startswith(('mlx_classifiers.vision', "
        "'mlx_vlm', 'PIL'))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
