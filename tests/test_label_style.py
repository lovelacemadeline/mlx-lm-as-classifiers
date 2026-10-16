"""Story 1.8 — option-text labels as a wire option (``options.label_style``)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlx_classifiers import FakeRunner, SystemOneRequest, decide
from mlx_classifiers.calibrate import gather, softmax
from mlx_classifiers.errors import LabelTokenizationError
from mlx_classifiers.prompt import NOUL_LABELS, compile_question
from mlx_classifiers.readout import bind_options

# ToyTokenizer keeps words of up to four letters whole, so these are single tokens.
MOVES = {"up": "Step north", "down": "Step south", "left": None, "wait": "Stay put"}


def _req(label_style="option-text", permutations=1, debug=False, **questions):
    questions = questions or {
        "move": {"type": "choice", "instructions": "Which action?", "criteria": MOVES}
    }
    options = {"permutations": permutations, "debug": debug}
    if label_style is not None:
        options["label_style"] = label_style
    return SystemOneRequest.model_validate(
        {
            "prefix": "You play a gridworld.",
            "state": "Enemy ahead.",
            "questions": questions,
            "options": options,
        }
    )


def _compile(req, name="move", runner=None, order=None):
    runner = runner or FakeRunner()
    return compile_question(
        req, name, template=runner.chat_template, tokenize=runner.tokenize, order=order
    )


def test_label_style_defaults_to_letters_and_is_closed():
    assert _req(label_style=None).effective_options.label_style == "letters"
    with pytest.raises(ValidationError, match="label_style"):
        _req(label_style="numbers")


def test_option_text_renders_names_and_asks_for_the_name():
    prompt = _compile(_req())
    assert prompt.label_style == "option-text"
    assert prompt.labels == tuple(MOVES)
    assert "- up: Step north\n- down: Step south\n- left\n- wait: Stay put" in prompt.text
    assert "A. " not in prompt.text and "letter" not in prompt.text
    assert "Answer with one option exactly as written." in prompt.text
    letters = _compile(_req(label_style="letters"))
    assert letters.labels == ("A", "B", "C", "D") and "A. up: Step north" in letters.text


def test_option_text_binds_the_names_as_first_token_labels():
    runner = FakeRunner()
    req = _req(debug=True)
    prompt = _compile(req, runner=runner)
    binding = bind_options(prompt, runner.tokenize)
    assert binding.labels == tuple(MOVES) and binding.tree.depth == 1
    after_marker = prompt.text[-1].isspace()  # the bare name after a newline, else " name"
    for name, tid in zip(MOVES, binding.token_ids, strict=True):
        assert [tid] == runner.tokenize(name if after_marker else " " + name)
    resp = decide(req, runner)
    ans = resp.answers["move"]
    assert ans.debug.option_token_ids == binding.option_token_ids
    row = runner.prefill(prompt.tokens).at(prompt.readout_position)
    expected = softmax(gather(row, binding.token_ids))
    assert list(ans.raw_probabilities.values()) == pytest.approx(expected)


def test_option_text_labels_follow_a_permuted_order():
    prompt = _compile(_req(), order=(2, 3, 0, 1))
    assert prompt.labels == ("left", "wait", "up", "down")
    assert prompt.text.index("- left") < prompt.text.index("- up")
    assert prompt.label_for(0) == "up"
    runner = FakeRunner()
    resp = decide(_req(permutations=4, debug=True), runner)
    ids = resp.answers["move"].debug.option_token_ids
    assert all(len(set(v)) == 1 and len(v) == 4 for v in ids.values())  # same id every order


def test_multi_token_name_raises_pointing_at_sequence_or_letters():
    crit = {"billing": None, "technical": None}  # both longer than one toy token
    req = _req(dept={"type": "choice", "instructions": "Team?", "criteria": crit})
    with pytest.raises(LabelTokenizationError, match="not single tokens") as e:
        decide(req, FakeRunner())
    assert 'options.readout = "sequence"' in str(e.value)
    assert 'options.label_style = "letters"' in str(e.value)
    assert (
        decide(_req(label_style="letters", dept=req.questions["dept"].model_dump()), FakeRunner())
        .answers["dept"]
        .choice
        in crit
    )


def test_noul_keeps_yes_no():
    req = _req(yn={"type": "noul", "instructions": "Is there an enemy?"})
    prompt = _compile(req, "yn")
    assert prompt.labels == NOUL_LABELS and "Answer Yes or No only." in prompt.text
    assert 0 <= decide(req, FakeRunner()).answers["yn"].noul <= 1


def test_score_uses_single_token_level_names_else_errors():
    good = _req(s={"type": "score", "instructions": "Danger?", "criteria": ["low", "mid", "high"]})
    prompt = _compile(good, "s")
    assert prompt.labels == ("low", "mid", "high") and "- low\n- mid\n- high" in prompt.text
    assert "Answer with one level exactly as written." in prompt.text
    resp = decide(good, FakeRunner())
    assert set(resp.answers["s"].probabilities) == {"0", "1", "2"}
    bad = _req(s={"type": "score", "instructions": "?", "criteria": ["trivial", "critical"]})
    with pytest.raises(LabelTokenizationError, match='label_style = "letters"'):
        decide(bad, FakeRunner())
