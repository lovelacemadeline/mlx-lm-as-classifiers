"""Story 1.4 — option binding and prefix-tree first-token readout (ADR-0003)."""

from __future__ import annotations

import string

import pytest
from conftest import hf_tokenize

from mlx_classifiers.errors import LabelTokenizationError, TooManyOptionsError
from mlx_classifiers.prompt import compile_question, compile_request
from mlx_classifiers.readout import PrefixTree, bind_labels, bind_options, emitted_label
from mlx_classifiers.runner import FakeRunner, ToyTokenizer
from mlx_classifiers.schema import SystemOneRequest
from mlx_classifiers.template import HFChatTemplate, ToyChatTemplate

CONTEXT = "<|bos|><|user|>\nPick one.<|end|>\n<|assistant|>\n"


def _req(**questions):
    return SystemOneRequest.model_validate({"state": "Payouts failing.", "questions": questions})


CHOICE = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None},
}


def test_single_token_labels_bind_at_depth_one():
    runner = FakeRunner()
    prompt = compile_request(_req(dept=CHOICE), runner)[0]
    binding = bind_options(prompt, runner.tokenize)
    assert binding.tree.depth == 1 and binding.tree.collisions == ()
    assert len(binding.token_ids) == len(set(binding.token_ids)) == 3  # one id per option
    assert binding.labels == ("A", "B", "C")
    assert binding.option_token_ids == {
        "billing": runner.tokenize("A"),
        "technical": runner.tokenize("B"),
        "sales": runner.tokenize("C"),
    }
    assert binding.readout_position == prompt.readout_position


def test_noul_binds_fixed_yes_no_pair():
    runner = FakeRunner()
    prompt = compile_request(_req(r={"type": "noul", "instructions": "Repro?"}), runner)[0]
    binding = bind_options(prompt, runner.tokenize)
    assert binding.options == ("yes", "no") and binding.labels == ("Yes", "No")


def test_shared_first_token_reports_depth_and_raises_naming_both():
    tok = ToyTokenizer()
    with pytest.raises(LabelTokenizationError) as exc:
        bind_labels(["billing", "billable", "sales"], CONTEXT, tok.encode)
    msg = str(exc.value)
    assert "'billing'" in msg and "'billable'" in msg and "'sales'" not in msg
    assert "depth 2" in msg and "sequence" in msg
    tree = PrefixTree.build({"billing": tok.encode("billing"), "billable": tok.encode("billable")})
    assert tree.depth == 2 and tree.collisions == (("billing", "billable"),)


def test_multi_token_label_is_a_hard_error():
    tok = ToyTokenizer()
    with pytest.raises(LabelTokenizationError, match="'technical' -> 3 tokens"):
        bind_labels(["sales", "technical"], CONTEXT, tok.encode)


def test_identical_label_tokens_have_no_depth():
    assert PrefixTree.build({"a": (5, 6), "b": (5, 6)}).depth is None
    assert PrefixTree.build({"a": (5,), "b": (5, 7)}).depth == 2


def test_leading_space_variant_matches_template_position():
    tok = ToyTokenizer()
    # After a newline the label is emitted bare ...
    ids, _ = bind_labels(["A", "B"], CONTEXT, tok.encode)
    assert ids == (tok.encode("A")[0], tok.encode("B")[0])
    # ... after "Answer:" it is emitted with a leading space.
    ids, _ = bind_labels(["A", "B"], CONTEXT + "Answer:", tok.encode)
    assert ids == (tok.encode(" A")[0], tok.encode(" B")[0])
    assert emitted_label("x:", "A") == " A" and emitted_label("x\n", "A") == "A"


def test_leading_space_variant_recorded_with_custom_template():
    class AnswerColon(ToyChatTemplate):
        generation_prompt = "<|assistant|>\nAnswer:"

    runner = FakeRunner(chat_template=AnswerColon())
    prompt = compile_request(_req(dept=CHOICE), runner)[0]
    binding = bind_options(prompt, runner.tokenize)
    assert binding.option_token_ids["billing"] == runner.tokenize(" A")


def test_label_merge_detected():
    # The context ends in a whitespace token; a label whose own leading whitespace
    # re-tokenizes that token cannot be read at the readout position.
    tok = ToyTokenizer()
    with pytest.raises(LabelTokenizationError, match="merges"):
        bind_labels([" A", " B"], "<|assistant|>\n ", tok.encode)


def test_more_than_26_options_fails_pointing_at_independent_scoring():
    criteria = {f"opt{i}": None for i in range(27)}
    req = _req(big={"type": "choice", "instructions": "?", "criteria": criteria})
    with pytest.raises(TooManyOptionsError, match="independent"):
        compile_request(req, FakeRunner())


def test_duplicate_labels_rejected():
    with pytest.raises(LabelTokenizationError, match="duplicate"):
        bind_labels(["A", "A"], CONTEXT, ToyTokenizer().encode)


# --- real tokenizers (skipped if not cached) ------------------------------------------------


def test_real_tokenizer_letters_and_yes_no_are_single_tokens(hf_tokenizer):
    template, tokenize = HFChatTemplate(hf_tokenizer), hf_tokenize(hf_tokenizer)
    letters = {f"o{i}": None for i in range(26)}
    req = _req(
        big={"type": "choice", "instructions": "?", "criteria": letters},
        yn={"type": "noul", "instructions": "?"},
    )
    for name in ("big", "yn"):
        prompt = compile_question(req, name, template=template, tokenize=tokenize)
        binding = bind_options(prompt, tokenize)
        assert binding.tree.depth == 1
        assert len(set(binding.token_ids)) == len(prompt.options)


def test_gemma_picks_bare_variant_after_turn_marker(gemma_tokenizer):
    tokenize = hf_tokenize(gemma_tokenizer)
    prompt = compile_question(
        _req(dept=CHOICE), "dept", template=HFChatTemplate(gemma_tokenizer), tokenize=tokenize
    )
    binding = bind_options(prompt, tokenize)
    bare, spaced = tokenize("A"), tokenize(" A")
    assert bare != spaced and len(bare) == len(spaced) == 1
    assert binding.option_token_ids["billing"] == bare  # after "<|turn>model\n": no space
    assert [gemma_tokenizer.decode([t]) for t in binding.token_ids] == list(
        string.ascii_uppercase[:3]
    )
