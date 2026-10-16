"""Decision log (story 3.3): one JSONL line per answered question, for later refits.

Every line is a record the calibrate CLI (story 3.2,
:mod:`mlx_classifiers.fitting`) accepts once a ``label`` field is added::

    {"v": 1, "ts": "2026-10-16T12:00:00.123+00:00", "request_hash": "sha256:...",
     "model": "<served model id>", "requested_model": "gemma-4-e2b-int8",
     "readout": "first-token", "permutations": 1, "calibrated": true,
     "temperature": 1.84, "question": "department", "type": "choice",
     "options": ["billing", "technical", "sales"],
     "raw_probabilities": [0.02, 0.96, 0.02], "probabilities": [0.08, 0.85, 0.07],
     "answer": "technical",
     "timing_ms": {"prefill": 31.2, "readout": 0.4, "total": 33.0}}

``options`` are the option names in declared order (criteria keys; ``"0"``..
for score levels; ``"yes"``/``"no"`` for noul) and both probability lists are
aligned with them; ``answer`` is the served choice, score index, or (noul)
``"yes"`` when calibrated P(yes) >= 0.5. ``temperature`` is the one applied (1 when uncalibrated).
``request_hash`` is shared by the lines of one request; ``timing_ms`` is that
request's. ``requested_model`` appears only when the request named one.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .prompt import question_options
from .schema import NoulAnswer, SystemOneRequest, SystemOneResponse

__all__ = ["DecisionLog", "decision_records", "request_hash"]

VERSION = 1


def request_hash(request: SystemOneRequest) -> str:
    """``sha256:`` of the request's canonical JSON (sorted keys, no whitespace)."""
    canon = json.dumps(request.to_wire(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


def decision_records(
    request: SystemOneRequest,
    response: SystemOneResponse,
    *,
    model: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """One log record per answered question of ``response``."""
    opts = request.effective_options
    base: dict[str, Any] = {
        "v": VERSION,
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "request_hash": request_hash(request),
        "model": model,
    }
    if request.model is not None:
        base["requested_model"] = request.model
    base |= {
        "readout": opts.readout,
        "permutations": opts.permutations,
        "calibrated": opts.calibrated,
        "temperature": temperature,
    }
    timing = response.timing_ms.to_wire() if response.timing_ms is not None else None
    out = []
    for name, ans in response.answers.items():
        options = list(question_options(request.questions[name]))
        if isinstance(ans, NoulAnswer):
            raw_yes = ans.raw_noul if ans.raw_noul is not None else ans.noul
            raw, cal = [raw_yes, 1 - raw_yes], [ans.noul, 1 - ans.noul]
            answer = "yes" if ans.noul >= 0.5 else "no"
        else:
            raw_map = ans.raw_probabilities or ans.probabilities
            raw = [raw_map[o] for o in options]
            cal = [ans.probabilities[o] for o in options]
            answer = ans.choice if ans.type == "choice" else ans.score
        out.append(
            {
                **base,
                "question": name,
                "type": ans.type,
                "options": options,
                "raw_probabilities": raw,
                "probabilities": cal,
                "answer": answer,
                "timing_ms": timing,
            }
        )
    return out


class DecisionLog:
    """Append-only JSONL file of decision records; safe to share across threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, records: list[dict[str, Any]]) -> None:
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(text)
