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


# --- story 6.3: the runner's image path ------------------------------------------------------

import base64

from mlx_classifiers import SystemOneRequest, decide
from mlx_classifiers.prompt import compile_question
from mlx_classifiers.prompt_cache import PromptCacheStore

COLOURS = {"red": (220, 20, 20), "green": (20, 180, 20), "blue": (20, 20, 220)}
QUESTION = {
    "type": "choice",
    "instructions": "What colour is the square in the picture?",
    "criteria": {"red": None, "green": None, "blue": None, "yellow": None},
}


def square_b64(rgb) -> str:
    rows = [
        [rgb if 16 <= x < 80 and 16 <= y < 80 else (255, 255, 255) for x in range(96)]
        for y in range(96)
    ]
    return base64.b64encode(write_png(96, 96, rows)).decode("ascii")


def colour_request(
    rgb, delta: bool = False, text: str = "Here is the picture:"
) -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {
            "prefix": "You look at pictures and answer questions.",
            "state": [text, {"type": "image/png", "base64": square_b64(rgb)}],
            "questions": {"q": QUESTION},
            "options": {"delta_cache": delta},
        }
    )


@pytest.fixture(scope="module")
def runner():
    from mlx_classifiers.mlx_runner import MLXRunner

    return MLXRunner.load(GEMMA)


def _raw(runner, req):
    return decide(req, runner).answers["q"].raw_probabilities


def test_text_only_decide_never_loads_the_vision_tower():
    from mlx_classifiers.mlx_runner import MLXRunner

    r = MLXRunner.load(GEMMA)
    assert r.image_tokens is not None and r.vision is None
    req = SystemOneRequest.model_validate({"state": "A red square.", "questions": {"q": QUESTION}})
    decide(req, r)
    assert r.vision is None and r.vision_encode_ms == []


def test_the_model_sees_the_image(runner):
    for name, rgb in COLOURS.items():
        raw = _raw(runner, colour_request(rgb))
        assert max(raw, key=raw.get) == name, raw


def test_image_request_is_deterministic(runner):
    req = colour_request(COLOURS["green"])
    runner.vision.clear()  # re-encode on the second call too
    assert _raw(runner, req) == _raw(runner, req)
    runner.vision.clear()
    assert _raw(runner, req) == _raw(runner, req)


def test_embedding_path_on_text_rows_equals_the_token_id_path(runner):
    """The vendored layer loop is mlx-lm's: on rows without an image it gives the
    same hidden states, bit for bit, as the backbone over token ids."""
    from mlx_lm.models.cache import make_prompt_cache

    from mlx_classifiers.vision.merge import forward_embeddings, merged_embeddings

    ids = mx.array(runner.tokenize("The quick brown fox jumps over the lazy dog. " * 4))
    a = runner._backbone(ids[None], cache=make_prompt_cache(runner.model))
    h = merged_embeddings(runner._backbone, ids, [])
    b = forward_embeddings(runner._backbone, h, make_prompt_cache(runner.model), ids)
    assert mx.array_equal(a, b).item()


def test_cache_only_chunks_split_inside_an_image_agree(runner, monkeypatch):
    """E2B is causal over image tokens, so a chunk boundary inside the run is exact
    up to bf16 chunking noise."""
    req = colour_request(COLOURS["blue"])
    whole = _raw(runner, req)
    monkeypatch.setattr(runner, "prefill_step", 64)
    chunked = _raw(runner, req)
    assert max(abs(whole[k] - chunked[k]) for k in whole) < 5e-3
    assert max(chunked, key=chunked.get) == "blue"


def test_delta_cache_keys_on_the_image_and_matches_uncached(runner, monkeypatch):
    monkeypatch.setattr(runner, "prompt_cache", PromptCacheStore())
    red, blue = (
        colour_request(COLOURS["red"], delta=True),
        colour_request(COLOURS["blue"], delta=True),
    )
    uncached_blue = _raw(runner, colour_request(COLOURS["blue"]))
    decide(red, runner)
    p = compile_question(
        blue,
        "q",
        template=runner.chat_template,
        tokenize=runner.tokenize,
        image_tokens=runner.image_tokens,
    )
    runner.prefill_calls.clear()
    out = decide(blue, runner).answers["q"].raw_probabilities
    assert runner.prefill_calls[0][0] == p.images[0].start  # hit stops at the first soft token
    assert max(abs(out[k] - uncached_blue[k]) for k in out) < 5e-3
    assert max(out, key=out.get) == "blue"
    runner.prefill_calls.clear()
    again = decide(blue, runner).answers["q"].raw_probabilities
    assert runner.prefill_calls[0][0] == p.readout_position  # same image: everything cached
    assert max(abs(again[k] - uncached_blue[k]) for k in out) < 5e-3


def test_int8_on_and_off_agree_on_image_requests(runner):
    from mlx_classifiers.mlx_runner import MLXRunner

    r8 = MLXRunner.load(GEMMA, int8=True)
    for name, rgb in COLOURS.items():
        stock, int8 = _raw(runner, colour_request(rgb)), _raw(r8, colour_request(rgb))
        assert max(int8, key=int8.get) == max(stock, key=stock.get) == name
        assert max(abs(stock[k] - int8[k]) for k in stock) < 5e-3


def test_vision_fast_is_bit_equal_to_the_vendored_path(tower):
    """Story 6.6: the compiled fast path reproduces the vendored tower on the GPU."""
    enc = tower.fast_encoder()
    assert enc.compile
    for colour in ((40, 160, 40), (200, 30, 30), (20, 20, 220)):
        png = frame_png(colour)
        slow = tower.encode_pixels(tower.pixel_values(png))[0]
        fast = enc.encode_png(png)
        assert fast.shape == slow.shape == (270, 1536) and fast.dtype == slow.dtype
        assert mx.array_equal(fast, slow).item()


@pytest.mark.parametrize(("budget", "n"), [(140, 120), (70, 63)])
def test_vision_fast_is_bit_equal_at_each_soft_token_budget(tower, budget, n):
    """Story 6.7: a smaller budget is a smaller pinned shape on the same fast path,
    still bit-equal to the vendored path at that budget."""
    tower.fast = True
    png = frame_png((20, 20, 220))
    slow = tower.encode_pixels(tower.pixel_values(png, budget))[0]
    fast = tower.encode_png(png, max_soft_tokens=budget)
    assert fast.shape == slow.shape == (n, 1536)
    assert mx.array_equal(fast, slow).item()
