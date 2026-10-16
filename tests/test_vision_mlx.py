"""Stories 6.2/6.3 — the vendored vision path on the real Gemma 4 E2B weights."""

from __future__ import annotations

import pytest
from conftest import GEMMA, HF_HUB

mx = pytest.importorskip("mlx.core")
pytest.importorskip("PIL")

from mlx_classifiers.images import write_png

_SNAP = HF_HUB / f"models--{GEMMA.replace('/', '--')}" / "snapshots"
pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        not any(_SNAP.glob("*/model.safetensors")), reason=f"{GEMMA} weights not in HF cache"
    ),
]


def frame_png(colour=(40, 160, 40)) -> bytes:
    """A gridworld-frame-shaped PNG: 160 x 192, a coloured square on grey."""
    rows = []
    for y in range(192):
        rows.append(
            [colour if 48 <= x < 112 and 48 <= y < 112 else (90, 90, 90) for x in range(160)]
        )
    return write_png(160, 192, rows)


@pytest.fixture(scope="module")
def tower():
    from mlx_lm.utils import _download

    from mlx_classifiers.vision import load_vision_tower

    return load_vision_tower(_download(GEMMA))


def test_tower_loads_every_vision_weight_and_quantizes_only_the_projector(tower):
    from mlx import nn

    assert isinstance(tower.embed_vision.embedding_projection, nn.QuantizedLinear)
    assert tower.embed_vision.embedding_projection.bits == 4
    q = tower.vision_tower.encoder.layers[0].self_attn.q_proj
    assert isinstance(q.linear, nn.Linear) and not isinstance(q.linear, nn.QuantizedLinear)
    assert float(q.input_max.item()) != float("inf")  # clip bounds came from the checkpoint
    assert len(tower.vision_tower.encoder.layers) == 16


def test_encode_a_frame_gives_270_soft_tokens_deterministically(tower):
    png = frame_png()
    tower.clear()
    a = tower.encode_png(png)
    b = tower.encode_pixels(tower.pixel_values(png))[0]
    assert a.shape == (270, 1536) and a.dtype == mx.bfloat16
    assert mx.array_equal(a, b).item()
