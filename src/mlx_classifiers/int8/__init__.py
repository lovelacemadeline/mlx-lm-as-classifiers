"""W8A8 INT8 prefill kernels for Gemma 4 on Apple M5 (story 2.3).

Vendored from ``aug/poorly-ported-gemma4-int8-optimisations/w8a8_wiring.py``;
see :mod:`mlx_classifiers.int8.wiring` for provenance. Importing this package
imports ``mlx``, so :class:`~mlx_classifiers.mlx_runner.MLXRunner` imports it
lazily, only when ``int8=True``.
"""

from .wiring import Int8LinearW8A8, patch_mlps, patch_model, patch_projs

__all__ = ["Int8LinearW8A8", "patch_mlps", "patch_model", "patch_projs"]
