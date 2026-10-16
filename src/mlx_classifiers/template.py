"""Chat templates: render a (system, user) pair with the assistant turn opened.

A :class:`ChatTemplate` returns the full prompt text and the text of its
cacheable prefix (the system turn). Runners expose one as ``chat_template`` so
prompt compilation needs nothing but a Runner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import PromptError

__all__ = ["ChatTemplate", "HFChatTemplate", "RenderedChat", "ToyChatTemplate"]


@dataclass(frozen=True)
class RenderedChat:
    """``text`` is the whole prompt ending in the open assistant turn;
    ``prefix_text`` is its cacheable head (empty when there is no system turn)."""

    prefix_text: str
    text: str

    def __post_init__(self) -> None:
        if not self.text.startswith(self.prefix_text):
            raise PromptError("rendered prompt does not start with its prefix text")


class ChatTemplate(Protocol):
    def render(self, system: str | None, user: str) -> RenderedChat: ...

    def render_messages(self, messages: Sequence[dict[str, str]]) -> RenderedChat:
        """A whole chat (``role``/``content`` dicts) with the assistant turn opened.
        ``prefix_text`` is the system turn exactly as :meth:`render` gives it, so
        a chat and a decision with the same system text share a prefix cache."""
        ...

    @property
    def generation_prompt(self) -> str:
        """The assistant-start marker text that ends every rendered prompt."""
        ...


class ToyChatTemplate:
    """The FakeRunner's template, in :class:`~mlx_classifiers.runner.ToyTokenizer` markup."""

    generation_prompt = "<|assistant|>\n"

    def render(self, system: str | None, user: str) -> RenderedChat:
        prefix = "<|bos|>" + (f"<|system|>\n{system}<|end|>\n" if system is not None else "")
        return RenderedChat(
            prefix_text=prefix if system is not None else "",
            text=f"{prefix}<|user|>\n{user}<|end|>\n{self.generation_prompt}",
        )

    def render_messages(self, messages: Sequence[dict[str, str]]) -> RenderedChat:
        msgs = list(messages)
        system = msgs.pop(0)["content"] if msgs and msgs[0]["role"] == "system" else None
        prefix = "<|bos|>" + (f"<|system|>\n{system}<|end|>\n" if system is not None else "")
        turns = "".join(f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in msgs)
        return RenderedChat(
            prefix_text=prefix if system is not None else "",
            text=f"{prefix}{turns}{self.generation_prompt}",
        )


class HFChatTemplate:
    """Adapter over any tokenizer exposing Hugging Face ``apply_chat_template``.

    Thinking modes are switched off (``enable_thinking=False``) so the token
    after the generation prompt is the answer itself.
    """

    def __init__(self, tokenizer: Any, **template_kwargs: Any):
        self.tokenizer = tokenizer
        self.template_kwargs = {"enable_thinking": False, **template_kwargs}

    def _apply(self, messages: list[dict[str, str]], generation: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation, **self.template_kwargs
        )

    @property
    def generation_prompt(self) -> str:
        msgs = [{"role": "user", "content": "x"}]
        closed, opened = self._apply(msgs, False), self._apply(msgs, True)
        if not opened.startswith(closed):
            raise PromptError("generation prompt is not a suffix of the closed conversation")
        return opened[len(closed) :]

    def render(self, system: str | None, user: str) -> RenderedChat:
        msgs = [] if system is None else [{"role": "system", "content": system}]
        text = self._apply([*msgs, {"role": "user", "content": user}], True)
        prefix = self._apply(msgs, False) if system is not None else ""
        if not text.startswith(prefix):
            raise PromptError("chat template renders the system turn differently in context")
        return RenderedChat(prefix_text=prefix, text=text)

    def render_messages(self, messages: Sequence[dict[str, str]]) -> RenderedChat:
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        text = self._apply(msgs, True)
        prefix = ""
        if msgs and msgs[0]["role"] == "system":
            prefix = self._apply(msgs[:1], False)
            if not text.startswith(prefix):
                prefix = ""  # template folds the system turn in; no shareable head
        return RenderedChat(prefix_text=prefix, text=text)
