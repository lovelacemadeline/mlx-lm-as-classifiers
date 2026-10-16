"""Story 6.6 — the ``vision_fast`` encode path equals the vendored (slow) path.

A tiny random-weight tower, on the **CPU** device only (no weights loaded, no GPU):
every stage of :class:`~mlx_classifiers.vision.fast.FastEncoder` against the
vendored modules, bit for bit, and the end-to-end soft tokens with and without
``mx.compile``. GPU parity on the real checkpoints is ``proto/vision_profile.py``'s
job (it prints max |d| per model)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("PIL")

from mlx.utils import tree_flatten

from mlx_classifiers.images import write_png
from mlx_classifiers.vision.encoder import one_hot
from mlx_classifiers.vision.tower import VisionTower


@pytest.fixture(autouse=True, scope="module")
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _config(head_dim=16, clip=True, standardize=False):
    heads = 2
    return {
        "vision_soft_tokens_per_image": 4,
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "head_dim": head_dim,
            "position_embedding_size": 64,
            "default_output_length": 4,
            "use_clipped_linears": clip,
            "standardize": standardize,
        },
    }


def _tower(seed=0, quantize=True, **kw) -> VisionTower:
    """Random bf16 weights (as the checkpoints are), finite clip bounds that bite,
    and a 4-bit projector like E2B / the 26B."""
    mx.random.seed(seed)
    tower = VisionTower(_config(**kw), text_hidden_size=64)
    new = []
    for name, p in tree_flatten(tower.parameters()):
        if name.endswith(("input_min", "output_min")):
            v = mx.array(-1.5)
        elif name.endswith(("input_max", "output_max")):
            v = mx.array(1.5)
        else:
            v = mx.random.normal(p.shape) * (0.3 if p.ndim > 1 else 1.0)
        new.append((name, v.astype(mx.bfloat16)))
    tower.load_weights(new, strict=True)
    if quantize:
        nn.quantize(
            tower.embed_vision,
            group_size=32,
            bits=4,
            class_predicate=lambda p, m: isinstance(m, nn.Linear),
        )
    mx.eval(tower.parameters())
    tower.eval()
    return tower


def _png(w=40, h=56, seed=1) -> bytes:
    rng = np.random.default_rng(seed)
    px = rng.integers(0, 256, size=(h, w, 3))
    return write_png(w, h, [[tuple(int(c) for c in px[y, x]) for x in range(w)] for y in range(h)])


def _slow(tower, png):
    return tower.encode_pixels(tower.pixel_values(png))[0]


def _equal(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def test_position_embedding_gather_equals_the_one_hot_matmul():
    tower = _tower()
    enc = tower.fast_encoder()
    vm = tower.vision_tower
    sh = enc.shape(96, 144)
    pos, pad, _ = vm._patch_positions_single(96, 144, max_patches=sh.ph * sh.pw)
    slow = vm.patch_embedder._position_embeddings(mx.array(pos[None]), mx.array(pad[None]))
    assert _equal(sh.pos_emb, slow)


def test_uint8_lookup_patchify_equals_the_float32_patchify():
    tower = _tower()
    enc = tower.fast_encoder()
    png = _png()
    pe = tower.vision_tower.patch_embedder
    pixels = tower.pixel_values(png)
    pos, pad, _ = tower.vision_tower._patch_positions_single(
        pixels.shape[2],
        pixels.shape[3],
        max_patches=(pixels.shape[2] // 16) * (pixels.shape[3] // 16),
    )
    slow = pe(pixels, mx.array(pos[None]), mx.array(pad[None]))
    image = mx.array(enc.uint8(png))
    assert _equal(enc.embed(image, enc.shape(*image.shape[:2])), slow)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"clip": False, "standardize": True},  # the 26B's switches
        {"head_dim": 12},  # padded to the fused SDPA width, like the 26B's 72 -> 80
    ],
    ids=["e2b-like", "26b-like", "padded-head"],
)
def test_fast_uncompiled_equals_slow_bit_for_bit(kw):
    tower = _tower(**kw)
    enc = tower.fast_encoder()
    enc.compile = False
    for png in (_png(), _png(56, 40, seed=2), _png(48, 48, seed=3)):
        assert _equal(enc.encode_png(png), _slow(tower, png))


def test_fast_compiled_equals_slow_bit_for_bit():
    tower = _tower(clip=True)
    enc = tower.fast_encoder()
    assert enc.compile
    png = _png()
    a = enc.encode_png(png)
    assert _equal(a, _slow(tower, png))
    assert _equal(enc.encode_png(png), a)  # second call: the cached trace


def test_one_trace_per_pinned_shape():
    tower = _tower()
    enc = tower.fast_encoder()
    key = enc.uint8(_png()).shape[:2]
    for seed in (1, 2, 3):
        enc.encode_png(_png(seed=seed))
    assert list(enc._shapes) == [key]
    fns = enc._shapes[key].fns
    assert fns is not None and len(fns) == 2
    enc.encode_png(_png(seed=4))
    assert enc._shapes[key].fns is fns


def test_vision_tower_fast_flag_memoises_and_matches_slow():
    slow_tower = _tower(seed=5)
    fast_tower = _tower(seed=5)
    fast_tower.fast = True
    png = _png(seed=7)
    a = fast_tower.encode_png(png)
    assert fast_tower.encodes == 1 and fast_tower.encode_png(png) is a
    assert _equal(a, slow_tower.encode_png(png))


def test_pool_weights_match_the_pooler():
    from mlx_classifiers.vision.fast import _pool_weights

    pos = mx.array(_tower().vision_tower._patch_positions_single(96, 144, 54)[0][None])
    w = _pool_weights(pos, 54, 6)
    assert w.shape == (1, 54, 6)
    assert bool(mx.all(w.sum(axis=1) == 1.0).item())
    assert _equal(w, one_hot(mx.argmax(w, axis=-1), 6) / 9)


def test_runner_vision_fast_defaults_on_and_the_tower_keeps_the_slow_reference():
    from mlx_classifiers.mlx_runner import MLXRunner

    assert MLXRunner.__dataclass_fields__["vision_fast"].default is True
    assert VisionTower(_config(), 64).fast is False  # the vendored path, unless asked
