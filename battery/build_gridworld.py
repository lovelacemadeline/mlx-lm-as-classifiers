"""Build ``battery/gridworld.jsonl`` and ``battery/images/``: the image slice (story 6.5).

Run from this repo's root with the *gridworld* repo's interpreter (it imports
``gridworld``; nothing here is imported by the package)::

    ../systemone-gridworld/.venv/bin/python battery/build_gridworld.py

Deterministic; the PNGs and the JSONL are committed. 30 frames: the oracle's
own trajectory on seeds 1..5, six consecutive ticks each (seed 1 from tick 6,
which covers its two attacks; seeds 2..5 from tick 0). Gold is
``gridworld.oracle.oracle`` of the frame. Every frame is asked three ways, as
one perturbation group (``group`` = the frame id, same gold):

- ``gridworld-text`` (``variant: base``): exactly what the gridworld's text
  mode sends (``backends.request.build_request(obs, mode="text")``);
- ``gridworld-image`` (``~p1``): the frame PNG with a fixed one-paragraph colour
  key and nothing else, so the map is only in the image;
- ``gridworld-text+image`` (``~p2``): exactly the gridworld's image mode (the
  same text plus the PNG, ``mode="image"``).

The PNG bytes are ``gridworld.render.image.to_png(obs, tile=16)`` (160 x 192 px,
270 Gemma 4 soft tokens), written to ``battery/images/<frame id>.png`` and
embedded base64 in the JSONL; ``tags.image_sha256`` pins them.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from gridworld import GridWorld
from gridworld.backends.request import build_request
from gridworld.oracle import oracle
from gridworld.render.image import to_png

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
OUT = ROOT / "gridworld.jsonl"

EPISODES = ((1, 6), (2, 0), (3, 0), (4, 0), (5, 0))  # (seed, first tick)
TICKS = 6
COLOUR_KEY = (
    "The picture is the map, north up: you are the blue square, the goal is the green "
    "cell, monsters are red cells (white pips are their hp), walls are dark grey and floor "
    "is light. The dark strip under the map shows your hp, the tick and your last action."
)


def frames():
    world = GridWorld()
    for seed, first in EPISODES:
        obs = world.reset(seed)
        for tick in range(first + TICKS):
            if tick >= first:
                yield f"gw-s{seed}-t{tick:02d}", obs
            result = world.step(oracle(obs))
            if result.done:
                raise RuntimeError(f"seed {seed} ended before tick {first + TICKS}")
            obs = result.observation


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    lines = []
    for frame_id, obs in frames():
        png = to_png(obs, 16)
        (IMAGES / f"{frame_id}.png").write_bytes(png)
        sha = hashlib.sha256(png).hexdigest()
        gold = {"action": oracle(obs).value}
        difficulty = "easy" if gold["action"] in ("north", "south", "east", "west") else "hard"
        text_req = build_request(obs, mode="text").request
        both_req = build_request(obs, mode="image").request
        block = both_req["state"][-1]
        image_req = {**text_req, "state": [COLOUR_KEY, block]}
        for variant, suffix, source, req in (
            ("base", "", "gridworld-text", text_req),
            ("p1", "~p1", "gridworld-image", image_req),
            ("p2", "~p2", "gridworld-text+image", both_req),
        ):
            tags = {"source": source, "difficulty": difficulty, "group": frame_id,
                    "variant": variant}
            if source != "gridworld-text":
                tags["image_sha256"] = sha
            lines.append({"id": frame_id + suffix, **req, "gold": gold, "tags": tags})
        assert base64.b64decode(block["base64"]) == png
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    print(f"wrote {len(lines)} cases ({len(lines) // 3} frames) to {OUT}")


if __name__ == "__main__":
    main()
