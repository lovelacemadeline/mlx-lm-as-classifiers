"""mlx-vlm reference for the story 6.4 parity test. Dev-only: imports mlx-vlm.

Run in-process by ``tests/test_vision_parity.py`` when mlx-vlm is importable (the
``vision-dev`` extra), else as a script under ``$MLX_VLM_PYTHON``::

    $MLX_VLM_PYTHON tests/vlm_reference.py REPO IDS.json IMAGE.png OUT.npz

Given the prompt token ids our compiler produced and the PNG, it saves mlx-vlm's
pixel values, projected soft tokens, merged input embeddings, per-layer inputs
and the full-forward logits at the last position, all as float32, plus the
mlx-vlm and mlx versions.
"""

from __future__ import annotations

import io
import json
import sys


def reference(repo: str, ids: list[int], png: bytes) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx_vlm import load
    from PIL import Image

    model, processor = load(repo)
    data, n_soft = processor.image_processor([Image.open(io.BytesIO(png))])
    pixels = mx.array(data["pixel_values"])
    input_ids = mx.array([ids])
    feats = model.encode_image(pixels)
    emb = model.get_input_embeddings(input_ids=input_ids, pixel_values=pixels)
    mm = (input_ids == model.config.image_token_id).astype(mx.int32)
    logits = model(input_ids, pixel_values=pixels, mm_token_type_ids=mm).logits[0, -1]
    f32 = lambda a: np.array(a.astype(mx.float32))
    from importlib.metadata import version

    return {
        "pixels": np.array(pixels),
        "n_soft": np.array(n_soft),
        "feats": f32(feats[0]),
        "embeds": f32(emb.inputs_embeds[0]),
        "per_layer": f32(emb.per_layer_inputs[0]) if emb.per_layer_inputs is not None else None,
        "logits": f32(logits),
        "versions": f"mlx-vlm {version('mlx-vlm')}, mlx {version('mlx')}",
    }


if __name__ == "__main__":
    import numpy as np

    repo, ids_path, png_path, out = sys.argv[1:5]
    with open(ids_path) as f:
        ids = json.load(f)
    with open(png_path, "rb") as f:
        ref = reference(repo, ids, f.read())
    np.savez(out, **{k: v for k, v in ref.items() if v is not None})
