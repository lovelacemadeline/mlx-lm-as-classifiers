---
type: epic
status: in-progress
title: Vision runner — image blocks in the state, vendored Gemma 4 vision path
epic: 6
owner: maddie
updated: 2026-10-16
---

# Epic 6 — Vision runner

Let a request's `state` carry image blocks beside text, and run them through
Gemma 4 E2B's vision path on the same split-prefill runner. Everything above
the runner (schema, labels, readout, calibration, server, fixtures, battery)
stays modality-blind (ADR-0002). The vision path is **vendored**, not taken as
a runtime dependency: the image encoder, the projector into the text embedding
space, and the merge at the image placeholder tokens, each with a provenance
header, the way the INT8 wiring was vendored. mlx-vlm is a dev-only reference
for a parity test and is never imported at runtime. Licence and any
non-standard ops are checked first; the dependency is the documented fallback.

## Stories
| id | title | status |
|---|---|---|
| 6.1 | Image blocks in the wire state and in prompt compilation | completed |
| 6.2 | Vendor the Gemma 4 vision encoder, projector and merge with provenance | completed |
| 6.3 | MLXRunner image path — encode, merge embeddings, same split prefill | completed |
| 6.4 | Parity test against mlx-vlm as a dev-only reference | completed |
| 6.5 | Image fixtures and a battery slice with images | pending |

## Related
- ADR-0002; [gridworld local-extensions](https://github.com/lovelacemadeline/systemone-gridworld/blob/main/schema/local-extensions.md) proposes the image-block shape (`{"type": "image/png", "base64": ...}` in the state list)
