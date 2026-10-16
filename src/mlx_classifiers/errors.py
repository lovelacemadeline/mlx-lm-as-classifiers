"""Errors raised while compiling a request. All are ``ValueError`` subclasses."""

from __future__ import annotations

__all__ = [
    "ClassifierError",
    "LabelTokenizationError",
    "OptionBindingError",
    "PromptError",
    "TooManyOptionsError",
]


class ClassifierError(ValueError):
    """Base class for request-compile errors."""


class PromptError(ClassifierError):
    """The chat template or tokenizer cannot give a stable prefix/readout split."""


class OptionBindingError(ClassifierError):
    """Options cannot be bound to labels."""


class TooManyOptionsError(OptionBindingError):
    """More options than single-letter labels (A..Z)."""


class LabelTokenizationError(OptionBindingError):
    """A label is not exactly one token at the readout position (ADR-0003)."""
