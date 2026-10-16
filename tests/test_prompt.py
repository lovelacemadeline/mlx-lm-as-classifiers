"""Story 1.3 — prompt compilation: chat template, prefix/state split, readout position."""

from __future__ import annotations

import json

from conftest import hf_tokenize

from mlx_classifiers.prompt import compile_question, compile_request, render_state
from mlx_classifiers.runner import FakeRunner
from mlx_classifiers.schema import SystemOneRequest
from mlx_classifiers.template import HFChatTemplate

PREFIX = "You are the triage router for a payments company."


def _request(state, prefix=PREFIX, **questions):
    questions = questions or {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, refunds",
                "technical": "Bugs, outages",
                "sales": None,
            },
        }
    }
    body = {"state": state, "questions": questions}
    if prefix is not None:
        body["prefix"] = prefix
    return SystemOneRequest.model_validate(body)


def test_prefix_tokens_are_a_stable_strict_prefix():
    runner = FakeRunner()
    a = compile_request(_request("Payouts failing for 3 days."), runner)[0]
    b = compile_request(_request({"ticket": 42, "body": "Refund please"}), runner)[0]
    assert 0 < a.prefix_len < len(a.tokens)
    assert a.prefix_tokens == tuple(
        runner.tokenize(runner.chat_template.render(PREFIX, "").prefix_text)
    )
    assert a.prefix_tokens == b.prefix_tokens  # byte-stable across different state
    assert a.tokens != b.tokens  # but the prompts differ


def test_no_prefix_means_nothing_cacheable():
    p = compile_request(_request("s", prefix=None), FakeRunner())[0]
    assert p.prefix_len == 0


def test_choice_user_turn_lists_labelled_options_and_reads_out_last_token():
    runner = FakeRunner()
    p = compile_request(_request("Payouts failing."), runner)[0]
    assert "A. billing: Payments, refunds\nB. technical: Bugs, outages\nC. sales\n" in p.text
    assert "Question: Which team should handle this?" in p.text
    assert p.readout_position == len(p.tokens) - 1
    assert p.text.endswith(runner.chat_template.generation_prompt)
    assert p.labels == ("A", "B", "C") and p.options == ("billing", "technical", "sales")


def test_score_and_noul_rendering():
    req = _request(
        "s",
        sev={"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]},
        rep={"type": "noul", "instructions": "Repro steps?"},
    )
    sev, rep = compile_request(req, FakeRunner())
    assert "Scale (lowest to highest):\nA. low\nB. high" in sev.text
    assert sev.options == ("0", "1")
    assert "Answer Yes or No only." in rep.text
    assert rep.options == ("yes", "no") and rep.labels == ("Yes", "No")


def test_json_state_is_sorted_pretty_json():
    state = {"zeta": 1, "alpha": {"b": 2, "a": [3, 1]}}
    rendered = render_state(state)
    assert rendered == json.dumps(state, indent=2, sort_keys=True)
    assert rendered.index('"alpha"') < rendered.index('"zeta"')
    assert rendered.index('"a"') < rendered.index('"b"')
    p1 = compile_request(_request(state), FakeRunner())[0]
    p2 = compile_request(_request({"alpha": {"a": [3, 1], "b": 2}, "zeta": 1}), FakeRunner())[0]
    assert p1.tokens == p2.tokens  # key order in the request does not matter


def test_list_state_is_numbered_blocks():
    assert render_state(["first", "second"]) == "[1] first\n\n[2] second"


def test_display_order_relabels_but_keeps_binding():
    req = _request("s")
    p = compile_question(
        req,
        "department",
        template=FakeRunner().chat_template,
        tokenize=FakeRunner().tokenize,
        order=(2, 0, 1),
    )
    assert "A. sales\nB. billing: Payments, refunds\nC. technical: Bugs, outages" in p.text
    assert p.label_for(0) == "B"  # billing is displayed second


# --- real chat templates (tokenizer only; skipped if not in the HF cache) -------------------


def test_real_template_readout_follows_assistant_marker(hf_tokenizer):
    template = HFChatTemplate(hf_tokenizer)
    tokenize = hf_tokenize(hf_tokenizer)
    p = compile_question(
        _request("Payouts failing."), "department", template=template, tokenize=tokenize
    )
    marker = tokenize(template.generation_prompt)
    assert marker, "template has no generation prompt"
    # The sequence ends with the assistant-start marker; the readout position is its
    # last token, so the next token predicted there is the answer label.
    assert p.tokens[-len(marker) :] == tuple(marker)
    assert p.readout_position == len(p.tokens) - 1
    assert "<think" not in p.text and "<|think|>" not in p.text  # thinking disabled
    # Natural tokenization equals the cached-prefix split (no boundary merge).
    assert p.tokens == tuple(tokenize(p.text))
    assert p.tokens[: p.prefix_len] == tuple(tokenize(template.render(PREFIX, "x").prefix_text))


def test_gemma_marker_is_turn_model(gemma_tokenizer):
    template = HFChatTemplate(gemma_tokenizer)
    assert template.generation_prompt == "<|turn>model\n"
    p = compile_question(
        _request("s"), "department", template=template, tokenize=hf_tokenize(gemma_tokenizer)
    )
    tail = [gemma_tokenizer.decode([t]) for t in p.tokens[-3:]]
    assert tail == ["<|turn>", "model", "\n"]


def test_real_template_prefix_stable_across_state(hf_tokenizer):
    template, tokenize = HFChatTemplate(hf_tokenizer), hf_tokenize(hf_tokenizer)
    a = compile_question(_request("one"), "department", template=template, tokenize=tokenize)
    b = compile_question(_request({"k": "two"}), "department", template=template, tokenize=tokenize)
    assert a.prefix_len > 0 and a.prefix_tokens == b.prefix_tokens
