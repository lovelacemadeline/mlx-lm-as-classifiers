---
type: story
status: pending
title: Vendor the Gemma 4 vision encoder, projector and merge with provenance
owner: maddie
updated: 2026-10-16
anchor: none  # outlined only; Epic 6 not started
---

# Story 6.2 — Vendor the Gemma 4 vision encoder, projector and merge with provenance

Copy the minimal modules from mlx-vlm (source repo, commit, licence, sha256 in a header) into `src/mlx_classifiers/vision/`: encoder, projector, merge. No mlx-vlm import at runtime. Weight loading maps the checkpoint's vision keys that gemma4.py currently drops. Check the licence is compatible and that every op is plain MLX.

## Acceptance Criteria
- **Given** the package installed without mlx-vlm **when** the vision path imports **then** it works; **given** mlx-vlm absent **when** text-only decide runs **then** nothing vision-related is loaded.
