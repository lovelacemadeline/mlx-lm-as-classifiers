"""Story 6.7 — the per-image soft-token budget (``vision_soft_tokens``: 70, 140, 280).

Weightless: the wire option, prompt compilation (run length, the budget on the
slot, the delta-cache key) and the fake runner. The tower half runs a tiny random
tower on the CPU, as ``test_vision_fast`` does."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from test_images import QUESTION, image_block

from mlx_classifiers.decide import decide
from mlx_classifiers.images import (
    DEFAULT_SOFT_TOKENS,
    SOFT_TOKEN_BUDGETS,
    ImageSlot,
    keyed_tokens,
    soft_token_count,
)
from mlx_classifiers.prompt import compile_question
from mlx_classifiers.prompt_cache import PromptCacheStore
from mlx_classifiers.runner import FakeRunner
from mlx_classifiers.schema import SystemOneRequest

FRAME = Path(__file__).resolve().parents[1] / "battery" / "images" / "gw-s3-t00.png"


def _request(png: bytes, budget: int | None = None) -> SystemOneRequest:
    body: dict = {"state": ["A frame:", image_block(png)], "questions": {"q": QUESTION}}
    if budget is not None:
        body["options"] = {"vision_soft_tokens": budget}
    return SystemOneRequest.model_validate(body)


def _compile(runner, req):
    return compile_question(
        req,
        "q",
        template=runner.chat_template,
        tokenize=runner.tokenize,
        image_tokens=runner.image_tokens,
    )


@pytest.mark.parametrize(
    ("budget", "size", "tokens"),
    [(280, (720, 864), 270), (140, (480, 576), 120), (70, (336, 432), 63)],
)
def test_the_gridworld_frame_resizes_and_pools_per_budget(budget, size, tokens):
    """160 x 192 -> at most budget x 9 patches of 16 px, sides multiples of 48,
    pooled 3 x 3: the patch count (tower sequence length) is tokens x 9."""
    from mlx_classifiers.images import resized_size

    assert resized_size(160, 192, 16, budget * 9, 3) == size
    assert soft_token_count(160, 192, 16, budget, 3) == tokens
    assert (size[0] // 16) * (size[1] // 16) == tokens * 9


def test_the_wire_option_takes_the_three_supported_budgets_only():
    png = FRAME.read_bytes()
    assert SOFT_TOKEN_BUDGETS == (70, 140, 280) and DEFAULT_SOFT_TOKENS == 280
    for b in SOFT_TOKEN_BUDGETS:
        assert _request(png, b).effective_options.vision_soft_tokens == b
    assert _request(png).effective_options.vision_soft_tokens is None
    for bad in (100, 560, 0):
        with pytest.raises(ValidationError):
            _request(png, bad)
    with pytest.raises(ValidationError):  # optional, not nullable
        SystemOneRequest.model_validate(
            _request(png).to_wire() | {"options": {"vision_soft_tokens": None}}
        )


@pytest.mark.parametrize(("budget", "tokens"), [(None, 270), (280, 270), (140, 120), (70, 63)])
def test_a_request_budget_sets_the_run_length_and_rides_on_the_slot(budget, tokens):
    runner = FakeRunner()
    (slot,) = _compile(runner, _request(FRAME.read_bytes(), budget)).images
    assert slot.length == tokens and slot.max_soft_tokens == (budget or 280)
    described = slot.describe()
    if budget in (None, 280):  # default fixtures keep their exact images record
        assert set(described) == {"start", "length", "sha256"}
    else:
        assert described["max_soft_tokens"] == budget


def test_the_same_png_at_two_budgets_never_shares_a_delta_key():
    png = FRAME.read_bytes()
    a = ImageSlot.from_png(0, 4, png, 280)
    b = ImageSlot.from_png(0, 4, png, 140)
    assert keyed_tokens((9,) * 4, [a])[0] != keyed_tokens((9,) * 4, [b])[0]
    assert keyed_tokens((9,) * 4, [a]) == keyed_tokens(
        (9,) * 4, [ImageSlot.from_png(0, 4, png, 280)]
    )


def test_delta_cache_hit_stops_at_the_image_when_only_the_budget_changes():
    runner = FakeRunner(prompt_cache=PromptCacheStore())
    png = FRAME.read_bytes()
    decide(_request(png, 280), runner)
    p = _compile(runner, _request(png, 140))
    runner.prefill_calls.clear()
    decide(_request(png, 140), runner)
    assert runner.prefill_calls[0][0] == p.images[0].start


def test_decide_answers_at_every_budget_on_the_fake_runner():
    png = FRAME.read_bytes()
    answers = {b: decide(_request(png, b), FakeRunner()).to_wire() for b in SOFT_TOKEN_BUDGETS}
    assert all(a["answers"]["q"]["choice"] in QUESTION["criteria"] for a in answers.values())


# --- tower (tiny random weights, CPU) ---------------------------------------------------


def test_tower_encodes_each_budget_to_its_own_count_and_memoises_per_budget():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("PIL")
    from test_vision_fast import _equal, _png, _slow, _tower

    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        tower = _tower()
        tower.fast = True
        png = _png(96, 96, seed=3)  # budget b: at most 9b patches, b tokens at most
        counts = {}
        for b in (4, 2):
            feats = tower.encode_png(png, max_soft_tokens=b)
            counts[b] = feats.shape[0]
            # the fast path at a budget equals the vendored path at that budget
            slow = tower.encode_pixels(tower.pixel_values(png, b))[0]
            assert _equal(feats, slow)
        assert counts[4] == 4 and counts[2] == 1 and tower.encodes == 2
        assert tower.encode_png(png, max_soft_tokens=4).shape[0] == 4 and tower.encodes == 2
        assert _equal(tower.encode_png(png), _slow(tower, png))  # default = the tower's own
    finally:
        mx.set_default_device(prev)


def test_runner_budget_field_validates_and_defaults_to_the_checkpoint():
    from mlx_classifiers.mlx_runner import MLXRunner

    assert MLXRunner.__dataclass_fields__["vision_soft_tokens"].default is None
    with pytest.raises(ValueError, match="not one of"):
        MLXRunner(model=None, tokenizer=None, chat_template=object(), vision_soft_tokens=100)
