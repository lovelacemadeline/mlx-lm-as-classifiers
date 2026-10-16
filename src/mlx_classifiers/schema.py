"""Pydantic v2 models for the /v1/systemone wire shape (ADR-0001).

The JSON Schema files in ``schema/`` are the contract; these models mirror them
field for field and ``tests/test_schema.py`` pins the two against drift. Model
class names match the ``$defs`` names in the schema files.

Serialise with :meth:`WireModel.to_wire`, which drops unset optional fields so
a parsed request re-serialises to the JSON it came from.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "Debug",
    "ImageBlock",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "RequestOptions",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "Timing",
    "Usage",
]


class WireModel(BaseModel):
    """Base for every wire model."""

    def to_wire(self) -> dict[str, Any]:
        """JSON-ready dict with unset optional fields omitted."""
        return self.model_dump(mode="json", exclude_unset=True)


class ClosedModel(WireModel):
    """A wire object with ``additionalProperties: false``."""

    model_config = ConfigDict(extra="forbid")


# --- request -----------------------------------------------------------------


class NoulQuestion(ClosedModel):
    type: Literal["noul"]
    instructions: str = Field(description="A yes/no question.")


class ChoiceQuestion(ClosedModel):
    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str | None] = Field(
        min_length=2, description="option name -> description (or null)."
    )


class ScoreQuestion(ClosedModel):
    type: Literal["score"]
    instructions: str
    criteria: list[str] = Field(
        min_length=2, description="Ordered levels, lowest first. Index is the score."
    )


class ImageBlock(ClosedModel):
    """x-local (story 6.1). An image in a list-valued ``state``, at its list position."""

    type: Literal["image/png"]
    base64: str = Field(description="The PNG file, base64-encoded.")


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class RequestOptions(ClosedModel):
    """x-local inference knobs; all optional."""

    permutations: int = Field(default=1, ge=1, le=8)
    calibrated: bool = True
    readout: Literal["first-token", "sequence"] = "first-token"
    label_style: Literal["letters", "option-text"] = "letters"
    delta_cache: bool = True
    debug: bool = False


class SystemOneRequest(ClosedModel):
    # Optional-but-not-nullable fields (the schema allows omission, not null):
    # annotated without ``| None`` so an explicit ``null`` is rejected; the
    # ``None`` default is never validated.
    model: str = Field(default=None, description="Model id. Omit for the server default.")
    state: str | dict[str, Any] | list[str | ImageBlock]
    prefix: str = Field(
        default=None, description="x-local. Stable rules/system text, KV-cached across requests."
    )
    questions: dict[str, Question] = Field(min_length=1)
    options: RequestOptions = Field(default=None)

    @property
    def effective_options(self) -> RequestOptions:
        return self.options if self.options is not None else RequestOptions()

    @property
    def has_images(self) -> bool:
        """True when the state list holds at least one image block (story 6.1)."""
        return isinstance(self.state, list) and any(isinstance(b, ImageBlock) for b in self.state)


# --- response ----------------------------------------------------------------


class Debug(WireModel):
    """x-local. Present when ``options.debug`` is true."""

    readout_position: int | None = None
    option_token_ids: dict[str, list[int]] | None = None
    raw_logits: dict[str, float] | None = None
    temperature: float | None = None
    permutation_spread: dict[str, float] | None = None


class NoulAnswer(WireModel):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1, description="Calibrated P(yes).")
    raw_noul: float | None = Field(default=None, ge=0, le=1)
    debug: Debug | None = None


class ChoiceAnswer(WireModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float] | None = None
    confidence: float = Field(ge=0, le=1)
    debug: Debug | None = None


class ScoreAnswer(WireModel):
    type: Literal["score"]
    score: int = Field(ge=0)
    expected_score: float | None = None
    legend: dict[str, str]
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float] | None = None
    confidence: float = Field(ge=0, le=1)
    debug: Debug | None = None


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(WireModel):
    input_tokens: int
    output_tokens: int
    cached_tokens: int | None = None


class Timing(WireModel):
    """x-local. Wall-clock breakdown in milliseconds."""

    prefill: float | None = None
    readout: float | None = None
    total: float | None = None


class SystemOneResponse(ClosedModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    timing_ms: Timing | None = None

    def to_wire(self) -> dict[str, Any]:
        """Responses are built, not parsed: emit every field we set, minus nulls."""
        return self.model_dump(mode="json", exclude_none=True)
