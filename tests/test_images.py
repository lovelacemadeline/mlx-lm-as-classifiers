"""Story 6.1 — image blocks in the wire state and in prompt compilation."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from conftest import GEMMA, HF_HUB, hf_tokenize, load_cached_tokenizer
from pydantic import ValidationError

from mlx_classifiers.decide import decide
from mlx_classifiers.errors import PromptError
from mlx_classifiers.fixtures import _with_label_style
from mlx_classifiers.images import (
    ImageSlot,
    ImageTokens,
    find_image_runs,
    image_key_id,
    image_tokens_from_config,
    keyed_tokens,
    png_size,
    soft_token_count,
    write_png,
)
from mlx_classifiers.prompt import compile_question, render_state
from mlx_classifiers.prompt_cache import PromptCacheStore
from mlx_classifiers.runner import FakeRunner
from mlx_classifiers.schema import ImageBlock, SystemOneRequest
from mlx_classifiers.template import HFChatTemplate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
QUESTION = {
    "type": "choice",
    "instructions": "What colour is the square?",
    "criteria": {"red": None, "green": None, "blue": None},
}


def square_png(rgb: tuple[int, int, int], size: int = 32, w: int | None = None) -> bytes:
    return write_png(w or size, size, [[rgb] * (w or size) for _ in range(size)])


def image_block(png: bytes) -> dict:
    return {"type": "image/png", "base64": base64.b64encode(png).decode("ascii")}


def image_request(
    *pngs: bytes, text: str = "A picture:", prefix: str | None = "Look closely."
) -> SystemOneRequest:
    body = {"state": [text, *map(image_block, pngs)], "questions": {"q": QUESTION}}
    if prefix is not None:
        body["prefix"] = prefix
    return SystemOneRequest.model_validate(body)


def _snapshot(repo: str) -> Path:
    snaps = sorted((HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"))
    snaps = [s for s in snaps if (s / "config.json").exists()]
    if not snaps:
        pytest.skip(f"{repo} not in the HF cache")
    return snaps[-1]


# --- schema ----------------------------------------------------------------------------------


def test_image_block_accepted_in_list_state_and_round_trips():
    png = square_png((255, 0, 0))
    wire = {"state": ["text", image_block(png)], "questions": {"q": QUESTION}}
    req = SystemOneRequest.model_validate(wire)
    assert isinstance(req.state[1], ImageBlock) and req.has_images
    assert req.to_wire() == wire


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image/jpeg", "base64": "AAAA"},
        {"type": "image/png"},
        {"type": "image/png", "base64": "AAAA", "url": "http://x"},
    ],
)
def test_bad_image_blocks_rejected(block):
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": ["t", block], "questions": {"q": QUESTION}})


def test_text_only_list_state_has_no_images():
    req = SystemOneRequest.model_validate({"state": ["a", "b"], "questions": {"q": QUESTION}})
    assert not req.has_images


# --- image arithmetic ------------------------------------------------------------------------


def test_png_writer_and_header_parse():
    png = square_png((1, 2, 3), size=20, w=30)
    assert png_size(png) == (30, 20)
    with pytest.raises(PromptError):
        png_size(b"GIF89a" + b"\0" * 30)


@pytest.mark.parametrize(
    ("height", "width", "n"),
    [
        (192, 160, 270),  # a 10x10 gridworld frame at tile 16: 864 x 720 (mlx-vlm 0.7.8 says 270)
        (100, 100, 256),  # square: 768 x 768
        (32, 32, 256),  # upscaled the same way
        (48, 2000, 216),  # very wide: 96 x 5184
        (480, 640, 266),  # 672 x 912
    ],
)
def test_soft_token_count_matches_the_gemma4_processor(height, width, n):
    """Expected values from mlx-vlm 0.7.8's ``get_aspect_ratio_preserving_size``."""
    assert soft_token_count(height, width) == n


def test_image_tokens_from_the_e2b_config():
    spec = image_tokens_from_config(_snapshot(GEMMA))
    assert spec is not None
    assert (spec.boi, spec.image, spec.eoi) == ("<|image>", "<|image|>", "<image|>")
    assert spec.image_token_id == 258880
    assert (spec.patch_size, spec.pooling_kernel_size, spec.max_soft_tokens) == (16, 3, 280)
    tok = load_cached_tokenizer(GEMMA)
    enc = hf_tokenize(tok)
    assert enc(spec.boi) == [255999] and enc(spec.image) == [258880] and enc(spec.eoi) == [258882]


def test_keyed_tokens_replace_runs_by_image_hash():
    a = ImageSlot.from_png(2, 3, square_png((255, 0, 0)))
    b = ImageSlot.from_png(2, 3, square_png((0, 0, 255)))
    toks = (1, 2, 9, 9, 9, 3)
    ka, kb = keyed_tokens(toks, [a]), keyed_tokens(toks, [b])
    assert ka[:2] == kb[:2] == (1, 2) and ka[5] == kb[5] == 3
    assert ka[2:5] == (image_key_id(a.sha256),) * 3 and ka[2] != kb[2] and ka[2] < 0
    assert keyed_tokens(toks) == toks
    assert find_image_runs(toks, 9) == [(2, 3)]


# --- prompt compilation ----------------------------------------------------------------------


def test_render_state_numbers_image_blocks_in_place():
    req = image_request(square_png((255, 0, 0)), text="before")
    assert render_state(req.state) == "[1] before\n\n[2] <image 0>"


def test_fake_runner_compiles_an_image_into_its_placeholder_run():
    runner = FakeRunner()
    png = square_png((255, 0, 0))
    p = compile_question(
        image_request(png),
        "q",
        template=runner.chat_template,
        tokenize=runner.tokenize,
        image_tokens=runner.image_tokens,
    )
    (slot,) = p.images
    spec = runner.image_tokens
    assert slot.length == 256 and slot.sha256 == hashlib.sha256(png).hexdigest()
    assert p.tokens[slot.start : slot.end] == (spec.image_token_id,) * 256
    assert p.tokens[slot.end] == runner.tokenize(spec.eoi)[0]
    assert slot.start > p.prefix_len and p.readout_position == len(p.tokens) - 1
    assert "[2] <|boi|><|image|>" in p.text


def test_gemma_compile_places_boi_n_image_tokens_eoi_at_the_block():
    tok = load_cached_tokenizer(GEMMA)
    spec = image_tokens_from_config(_snapshot(GEMMA))
    frame = square_png((0, 128, 0), size=192, w=160)  # gridworld frame shape: 270 soft tokens
    req = image_request(frame, square_png((255, 0, 0)), text="Two pictures follow.")
    p = compile_question(
        req, "q", template=HFChatTemplate(tok), tokenize=hf_tokenize(tok), image_tokens=spec
    )
    assert [s.length for s in p.images] == [270, 256]
    for s in p.images:
        assert p.tokens[s.start - 1] == 255999 and p.tokens[s.end] == 258882
        assert set(p.tokens[s.start : s.end]) == {258880}
    first, second = p.images
    assert first.end < second.start
    # the text around the run is the text-only rendering's, block for block
    head = p.text[: p.text.index("<|image>")]
    assert head.endswith("State:\n[1] Two pictures follow.\n\n[2] ")
    assert p.text.endswith("<turn|>\n<|turn>model\n")
    assert p.readout_position == len(p.tokens) - 1


def test_text_only_requests_compile_byte_identically_to_the_pinned_fixtures():
    """Every seed fixture (text-only) recompiles to its pinned tokens with the
    vision-aware compiler and an image convention present."""
    tok = load_cached_tokenizer(GEMMA)
    spec = image_tokens_from_config(_snapshot(GEMMA))
    template, enc = HFChatTemplate(tok), hf_tokenize(tok)
    paths = sorted(FIXTURES.glob("*.json"))
    assert paths
    for path in paths:
        fx = json.loads(path.read_text())
        if (
            fx["request"]["state"]
            and isinstance(fx["request"]["state"], list)
            and any(isinstance(b, dict) for b in fx["request"]["state"])
        ):
            continue  # image fixtures (story 6.5) are pinned by their own test
        req = _with_label_style(SystemOneRequest.model_validate(fx["request"]), fx["label_style"])
        for image_tokens in (None, spec):
            p = compile_question(
                req, fx["question"], template=template, tokenize=enc, image_tokens=image_tokens
            )
            assert list(p.tokens) == fx["expected"]["tokens"], path.name
            assert p.images == ()


def test_image_state_needs_a_vision_runner():
    runner = FakeRunner()
    with pytest.raises(PromptError, match="no vision path"):
        compile_question(
            image_request(square_png((1, 1, 1))),
            "q",
            template=runner.chat_template,
            tokenize=runner.tokenize,
            image_tokens=None,
        )


def test_bad_png_payload_is_a_prompt_error():
    runner = FakeRunner()
    req = SystemOneRequest.model_validate(
        {
            "state": ["t", {"type": "image/png", "base64": "bm90IGEgcG5n"}],
            "questions": {"q": QUESTION},
        }
    )
    with pytest.raises(PromptError, match="not a PNG"):
        compile_question(
            req,
            "q",
            template=runner.chat_template,
            tokenize=runner.tokenize,
            image_tokens=runner.image_tokens,
        )


def test_literal_placeholder_text_in_the_state_is_refused():
    tok = load_cached_tokenizer(GEMMA)
    spec = image_tokens_from_config(_snapshot(GEMMA))
    req = SystemOneRequest.model_validate(
        {"state": ["<|image|>", image_block(square_png((1, 1, 1)))], "questions": {"q": QUESTION}}
    )
    with pytest.raises(PromptError, match="image-token runs"):
        compile_question(
            req, "q", template=HFChatTemplate(tok), tokenize=hf_tokenize(tok), image_tokens=spec
        )


# --- through decide on the fake runner -------------------------------------------------------


def test_decide_with_images_depends_on_the_image_and_is_deterministic():
    runner = FakeRunner()
    red, blue = square_png((255, 0, 0)), square_png((0, 0, 255))
    a1 = decide(image_request(red), runner).to_wire()["answers"]
    a2 = decide(image_request(red), runner).to_wire()["answers"]
    b = decide(image_request(blue), runner).to_wire()["answers"]
    assert a1 == a2
    assert a1["q"]["raw_probabilities"] != b["q"]["raw_probabilities"]


def test_delta_cache_does_not_share_kv_across_different_images():
    runner = FakeRunner(prompt_cache=PromptCacheStore())
    red, blue = square_png((255, 0, 0)), square_png((0, 0, 255))
    decide(image_request(red), runner)
    p = compile_question(
        image_request(blue),
        "q",
        template=runner.chat_template,
        tokenize=runner.tokenize,
        image_tokens=runner.image_tokens,
    )
    runner.prefill_calls.clear()
    decide(image_request(blue), runner)
    (cached, _), *_ = runner.prefill_calls
    assert cached == p.images[0].start  # the hit stops at the first soft token
    runner.prefill_calls.clear()
    decide(image_request(blue), runner)
    assert runner.prefill_calls[0][0] == p.readout_position  # same image: all but the readout


def test_sequence_mode_carries_images_into_the_prefix_cache():
    runner = FakeRunner()
    body = image_request(square_png((255, 0, 0))).to_wire() | {"options": {"readout": "sequence"}}
    out = decide(SystemOneRequest.model_validate(body), runner)
    assert out.answers["q"].choice in QUESTION["criteria"]


def test_image_tokens_placeholder_text():
    spec = ImageTokens(boi="<b>", image="<i>", eoi="<e>", image_token_id=7)
    png = square_png((0, 0, 0), size=48, w=96)
    n = spec.soft_tokens(png)
    assert spec.placeholder(png) == "<b>" + "<i>" * n + "<e>"


def test_chunk_bounds_move_off_image_runs_for_bidirectional_models():
    from mlx_classifiers.mlx_runner import _bounds_off_runs

    assert _bounds_off_runs([0, 64, 128, 192], [(50, 100)]) == [0, 50, 128, 192]
    assert _bounds_off_runs([0, 64, 128], [(0, 100)]) == [0, 100, 128]  # run at the start
    assert _bounds_off_runs([0, 10, 20], []) == [0, 10, 20]
    assert _bounds_off_runs([0, 64, 96, 128], [(40, 120)]) == [0, 40, 120, 128]
