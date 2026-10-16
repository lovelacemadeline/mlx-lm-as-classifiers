---
type: story
status: completed
title: Image blocks in the wire state and in prompt compilation
owner: maddie
updated: 2026-10-16
anchor:
  - symbol: "src/mlx_classifiers/schema.py::ImageBlock"
  - symbol: "src/mlx_classifiers/schema.py::SystemOneRequest.has_images"
  - symbol: "src/mlx_classifiers/images.py::ImageTokens"
  - symbol: "src/mlx_classifiers/images.py::ImageSlot"
  - symbol: "src/mlx_classifiers/images.py::soft_token_count"
  - symbol: "src/mlx_classifiers/images.py::image_tokens_from_config"
  - symbol: "src/mlx_classifiers/images.py::keyed_tokens"
  - symbol: "src/mlx_classifiers/prompt.py::compile_question"
  - symbol: "src/mlx_classifiers/prompt.py::render_state"
  - symbol: "src/mlx_classifiers/runner.py::check_prefill_args"
  - symbol: "src/mlx_classifiers/runner.py::FakeRunner.prefill"
  - test: "tests/test_schema.py::test_pydantic_schema_matches_json_schema_files"
  - test: "tests/test_images.py::test_image_block_accepted_in_list_state_and_round_trips"
  - test: "tests/test_images.py::test_bad_image_blocks_rejected"
  - test: "tests/test_images.py::test_soft_token_count_matches_the_gemma4_processor"
  - test: "tests/test_images.py::test_image_tokens_from_the_e2b_config"
  - test: "tests/test_images.py::test_gemma_compile_places_boi_n_image_tokens_eoi_at_the_block"
  - test: "tests/test_images.py::test_text_only_requests_compile_byte_identically_to_the_pinned_fixtures"
  - test: "tests/test_images.py::test_literal_placeholder_text_in_the_state_is_refused"
  - test: "tests/test_images.py::test_delta_cache_does_not_share_kv_across_different_images"
---

# Story 6.1 — Image blocks in the wire state and in prompt compilation

A list-valued `state` may now hold `{"type": "image/png", "base64": ...}`
blocks beside strings: additive and x-local, the schema stays v1. The block
shape is the one the gridworld proposed
(`systemone-gridworld/schema/local-extensions.md`, `backends/request.py::build_state`:
`[<state text>, {"type": "image/png", "base64": <PNG>}]`), field for field
(`additionalProperties: false`), so that repo can drop its in-memory schema
extension once it re-vendors ours.

## What compiles

A list state renders as before, `[i] block` joined by blank lines; an image
block renders as `[i] ` followed by the model's **placeholder run** at that
position. Compilation renders the user turn with a private-use sentinel per
image, lets the chat template render, swaps each sentinel for the run text,
then tokenizes the whole prompt once (the same string replacement the Hugging
Face / mlx-vlm processor does on the template's `<|image|>`). The runs are
then located in the token ids and must match the image blocks one to one, else
`PromptError` (e.g. a state whose text itself contains `<|image|>`).
`CompiledPrompt.images` records one `ImageSlot` per image: run start, run
length, sha256 of the PNG bytes (the bytes ride along for the runner).

### Gemma 4's placeholder run (verified, not guessed)

From the E2B snapshot (`42f62737…`) and mlx-vlm 0.7.8's processor:

| piece | token | id | source |
|---|---|---|---|
| boi | `<|image>` | 255999 | `config.json: boi_token_id`, `tokenizer_config.json: boi_token` |
| soft token, N times | `<|image|>` | 258880 | `config.json: image_token_id` |
| eoi | `<image|>` | 258882 | `config.json: eoi_token_id` |

**N is per image, not a constant.** The processor resizes (aspect ratio kept,
bicubic) to the largest size with both sides multiples of
`patch_size * pooling_kernel_size` = 48 px holding at most
`max_soft_tokens * 3^2` = 2520 patches of 16 px, and N = patches / 9. So
`vision_soft_tokens_per_image: 280` is a ceiling. A 160 x 192 gridworld frame
goes to 720 x 864 and **N = 270** (confirmed by running mlx-vlm on that frame:
285 prompt ids, 270 image tokens, boi at 8, eoi at 279); any square image is
768 x 768 and N = 256. `images.soft_token_count` vendors the arithmetic
(`get_aspect_ratio_preserving_size`, MIT, sha256 in its docstring);
`images.image_tokens_from_config` reads all of the above from the snapshot's
JSON, no weights. The 26B-A4B uses the same ids and processor numbers.

## The runner seam

Images are an optional Runner extension (as delta caching is): a vision
runner has `image_tokens`, and `prefill` / `cache_prefix` take
`images: Sequence[ImageSlot]`. `decide`, sequence scoring and the no-delta
proxy pass them through; the server's prefix LRU is bypassed for an
image-carrying `cache_prefix` (it is keyed by token ids alone). A
`PrefixCache` remembers its images and `check_prefill_args` refuses a prefix
cache whose images differ. The delta cache keys on `keyed_tokens`: the run's
ids replaced by a negative id from the image's sha256, so prompts differing
only in their image diverge at the first soft token. The FakeRunner has a toy
convention (`<|boi|>`, `<|image|>`, `<|eoi|>`, Gemma's N) and mixes the image
hashes into its logits, so weightless tests see images matter.

A text-only request takes none of these paths: the 8 pinned seed fixtures
recompile to their exact token ids with and without an image convention
present (test below), and the suite's existing 171 tests are unchanged.

## Acceptance Criteria
- **Given** a state with one text and one image block **when** compiled **then** the token sequence contains `boi + N * image + eoi` at the image block's position (N from the processor arithmetic, 270 for a gridworld frame), the readout position is still the last token, and the image's sha256 is recorded. *(test_gemma_compile_places_boi_n_image_tokens_eoi_at_the_block, test_fake_runner_compiles_an_image_into_its_placeholder_run)*
- **Given** a text-only request **when** compiled **then** its token ids are byte-identical to the pinned fixtures. *(test_text_only_requests_compile_byte_identically_to_the_pinned_fixtures)*
- **Given** the request schema file and the pydantic models **when** compared **then** they agree (drift test), and a malformed block (other type, missing `base64`, extra keys) is rejected. *(test_pydantic_schema_matches_json_schema_files, test_bad_image_blocks_rejected)*
- **Given** two prompts that differ only in the image **when** the delta cache is on **then** the second reuses KV only up to the image's first soft token. *(test_delta_cache_does_not_share_kv_across_different_images)*
