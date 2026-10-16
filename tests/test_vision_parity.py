"""Story 6.4 — parity of the vendored vision path against mlx-vlm (dev-only reference).

Runs when mlx-vlm is importable (``pip install mlx-classifiers[vision-dev]``) or
when ``MLX_VLM_PYTHON`` names an interpreter that has it (mlx-vlm 0.7.8 needs
mlx >= 0.32.3, so a separate venv keeps this repo's mlx 0.32.0 pin); skipped
otherwise. Same PNG and the same prompt token ids through both paths:

- pixel values and projected soft tokens: exact;
- merged input embeddings and per-layer inputs: max abs difference at most
  ``EMBED_TOL`` (one bf16 step at the largest embedding magnitude; measured 0);
- full-forward logits at the readout position vs our split prefill: same
  argmax over the option labels and over the vocab, option-label logits within
  ``LOGIT_TOL`` (the cross-path bf16 noise of story 2.5).
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import GEMMA, HF_HUB

mx = pytest.importorskip("mlx.core")
pytest.importorskip("PIL")

from mlx_classifiers import SystemOneRequest
from mlx_classifiers.calibrate import gather
from mlx_classifiers.images import write_png
from mlx_classifiers.prompt import compile_question
from mlx_classifiers.readout import bind_options

EMBED_TOL = 0.0625  # bf16 has 8 significant bits: one step at |x| in [8, 16)
LOGIT_TOL = 0.2

MOE = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"


def _cached(repo: str) -> bool:
    snap = HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    return any(snap.glob("*/model*.safetensors"))


_IN_PROCESS = importlib.util.find_spec("mlx_vlm") is not None
_EXTERNAL = os.environ.get("MLX_VLM_PYTHON")
pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        not (_IN_PROCESS or _EXTERNAL),
        reason="mlx-vlm absent (install the vision-dev extra or set MLX_VLM_PYTHON)",
    ),
]

QUESTION = {
    "type": "choice",
    "instructions": "Where is the blue square relative to the green one?",
    "criteria": {"left": None, "right": None, "above": None, "below": None},
}


def frame() -> bytes:
    """160 x 192 (a gridworld frame's size): a blue and a green square on light grey."""
    rows = []
    for y in range(192):
        row = []
        for x in range(160):
            if 16 <= x < 48 and 64 <= y < 96:
                row.append((40, 92, 220))
            elif 112 <= x < 144 and 64 <= y < 96:
                row.append((60, 176, 80))
            else:
                row.append((232, 230, 218))
        rows.append(row)
    return write_png(160, 192, rows)


def _reference(repo: str, ids: list[int], png: bytes, tmp_path: Path) -> dict:
    if _IN_PROCESS:
        sys.path.insert(0, str(Path(__file__).parent))
        from vlm_reference import reference

        return reference(repo, ids, png)
    ids_path, png_path, out = tmp_path / "ids.json", tmp_path / "f.png", tmp_path / "ref.npz"
    ids_path.write_text(json.dumps(ids))
    png_path.write_bytes(png)
    script = Path(__file__).parent / "vlm_reference.py"
    run = subprocess.run(
        [_EXTERNAL, "-I", str(script), repo, str(ids_path), str(png_path), str(out)],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    with np.load(out) as z:
        return {k: z[k] for k in z.files} | {"versions": "external"}


@pytest.mark.parametrize("repo", [GEMMA, MOE], ids=["e2b", "26b"])
def test_merged_embeddings_and_readout_match_mlx_vlm(repo, tmp_path):
    """E2B: causal image tokens, per-layer inputs. 26B-A4B: bidirectional image
    blocks (the mask overlay), standardized tower, no per-layer inputs."""
    if not _cached(repo):
        pytest.skip(f"{repo} weights not in HF cache")
    from mlx_classifiers.mlx_runner import MLXRunner
    from mlx_classifiers.vision.merge import merged_embeddings, per_layer_token_ids

    png = frame()
    req = SystemOneRequest.model_validate(
        {
            "prefix": "You look at pictures and answer questions.",
            "state": [
                "Here is the picture:",
                {"type": "image/png", "base64": base64.b64encode(png).decode()},
            ],
            "questions": {"q": QUESTION},
        }
    )
    runner = MLXRunner.load(repo)
    p = compile_question(
        req,
        "q",
        template=runner.chat_template,
        tokenize=runner.tokenize,
        image_tokens=runner.image_tokens,
    )
    ref = _reference(repo, list(p.tokens), png, tmp_path)  # before our weights grow the heap

    tower = runner.load_vision()
    pixels = np.array(tower.pixel_values(png))
    assert np.array_equal(pixels, ref["pixels"])
    (slot,) = p.images
    feats = runner._image_features(slot)
    assert np.array_equal(np.array(feats.astype(mx.float32)), ref["feats"])

    ids = mx.array(p.tokens)
    pieces = [(slot.start, slot.length, feats)]
    ours = np.array(merged_embeddings(runner._backbone, ids, pieces)[0].astype(mx.float32))
    d_embed = float(np.abs(ours - ref["embeds"]).max())
    d_pl = 0.0
    if runner._backbone.hidden_size_per_layer_input:
        raw = runner._backbone._get_per_layer_inputs(per_layer_token_ids(ids, pieces)[None])
        d_pl = float(np.abs(np.array(raw[0].astype(mx.float32)) - ref["per_layer"]).max())
    print(f"[6.4] {ref['versions']}: max|d| embeds {d_embed:.3g}, per-layer {d_pl:.3g}")
    assert d_embed <= EMBED_TOL and d_pl <= EMBED_TOL

    binding = bind_options(p, runner.tokenize)
    r = runner.prefill(p.tokens, None, [p.readout_position], images=p.images, delta=False)
    row = r.at(p.readout_position)
    ours_opt = gather(row, binding.token_ids)
    ref_opt = [float(ref["logits"][t]) for t in binding.token_ids]
    print(f"[6.4] option logits ours {np.round(ours_opt, 3)} mlx-vlm {np.round(ref_opt, 3)}")
    assert int(np.argmax(ours_opt)) == int(np.argmax(ref_opt))
    assert max(abs(a - b) for a, b in zip(ours_opt, ref_opt, strict=True)) <= LOGIT_TOL
