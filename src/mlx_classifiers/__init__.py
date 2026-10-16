"""Typed, calibrated decisions from an MLX-LM decoder in one forward pass."""

from .decide import decide
from .runner import FakeRunner, Runner
from .schema import SystemOneRequest, SystemOneResponse

__version__ = "0.0.1"
__all__ = ["FakeRunner", "Runner", "SystemOneRequest", "SystemOneResponse", "decide"]
