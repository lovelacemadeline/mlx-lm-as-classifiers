---
type: guide
status: active
title: mlx-lm-as-classifiers — README
owner: maddie
updated: 2026-10-16
---

# mlx-lm-as-classifiers

Typed, calibrated decisions from an MLX-LM decoder in one forward pass. You
declare the questions (`noul` yes/no, `choice` enum, `score` ordered scale)
and pass the program state; you get back probabilities over exactly the
options you declared. There is no decode loop: the logits at the last prefill
position are the decision. The wire shape is compatible with TypeSafe Jev's
`POST /v1/systemone`, with a few documented extensions.

What's here, all on Apple silicon:

- **The library**: schema models, prompt compilation, single-token label
  binding (letters or the option names), first-token and sequence readout,
  temperature calibration, permutation averaging, and a runner for Gemma 4
  that prefills cache-only and runs the KV-consumer layers on the readout
  row only.
- **Delta prefix caching** on the longest common token prefix, so a long
  state that changes a little per call costs a little per call.
- **INT8 paths** for Gemma 4 E2B (dense MLP and projections) and the 26B MoE
  (experts), vendored with provenance, behind one flag. M5 only.
- **A server**: `/v1/systemone` and `/v1/chat/completions` from one loaded
  model with a shared prefix cache, plus a calibrate CLI and a decision log.
- **Fixtures** pinning token ids and probabilities for ports to reproduce,
  and a **battery** of 343 labelled decisions reporting accuracy, ECE,
  position bias, and prefix-perturbation flip rate.

## Quick start

Requires Python 3.12+, Apple silicon, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lovelacemadeline/mlx-lm-as-classifiers
cd mlx-lm-as-classifiers
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "mlx==0.32.0" "mlx-lm==0.31.3" -e '.[server,dev]'

# serve decisions and chat from one model (downloads Gemma 4 E2B on first run)
.venv/bin/mlx-classifiers serve --port 8080

curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "prefix": "You are a triage router.",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages", "sales": null}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"}
  }
}'
```

Other entry points: `mlx-classifiers bench` (decisions per second against
state length, cache and INT8 on or off), `battery`, `calibrate`, and
`fixtures generate|check`. Pin `mlx==0.32.0` and `mlx-lm==0.31.3`: the MoE
kernel checks itself against that MLX version and disables itself otherwise.

The sequential benchmark that drives all of this, a gridworld you can watch
a model play, lives at
[systemone-gridworld](https://github.com/lovelacemadeline/systemone-gridworld).

Docs follow [`docs/doc-conventions.md`](docs/doc-conventions.md) and are
checked by `docloom check`. Start at
[`docs/product/prd-system-one-classifier.md`](docs/product/prd-system-one-classifier.md);
`docs/reference/wire-schema.md` is the API. Licence: MIT.
