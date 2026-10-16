"""One process, one loaded model: ``POST /v1/systemone`` beside ``POST /v1/chat/completions``.

Both routes run on the same :class:`~mlx_classifiers.runner.Runner` and draw
their system-turn KV from one :class:`PrefixLRU`, keyed by the prefix token ids.
A chat completion and a decision whose system text matches therefore share a
single prefilled prefix (the chat template renders the system turn identically
for both: :meth:`~mlx_classifiers.template.ChatTemplate.render_messages`).

Every model call runs on one dedicated worker thread (the one that loaded the
model: MLX streams are per-thread), which also serialises them: one model, one
GPU.

``litestar`` and ``uvicorn`` are the ``server`` extra; ``mlx_lm`` is only needed
for the chat route's generation (non-streaming, enough to prove the
single-process story).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litestar import Litestar, Request, Response, post
from litestar.datastructures import State
from litestar.exceptions import SerializationException
from pydantic import ValidationError

from .decide import decide
from .decision_log import DecisionLog, decision_records
from .errors import ClassifierError
from .fitting import CalibrationConfig, default_calibration_path
from .runner import PrefixCache, Runner
from .schema import SystemOneRequest

__all__ = [
    "CachingRunner",
    "ChatResult",
    "PrefixLRU",
    "ServerState",
    "build_app_from_args",
    "create_app",
    "main",
    "mlx_generate",
]

#: ``generate(runner, tokens, prefix_cache, max_tokens, temperature) -> ChatResult``
Generate = Callable[[Any, Sequence[int], PrefixCache | None, int, float], "ChatResult"]


class PrefixLRU:
    """Least-recently-used prefix caches keyed by their token ids.

    ``get`` returns ``(cache, hit)``; a miss prefills via ``build`` (a runner's
    ``cache_prefix``) and may evict the oldest entry. Thread-safe.
    """

    def __init__(self, build: Callable[[Sequence[int]], PrefixCache], capacity: int = 16):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.build, self.capacity = build, capacity
        self._entries: OrderedDict[tuple[int, ...], PrefixCache] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, tokens: Sequence[int]) -> bool:
        return tuple(tokens) in self._entries

    def get(self, tokens: Sequence[int]) -> tuple[PrefixCache, bool]:
        key = tuple(tokens)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key], True
        cache = self.build(key)
        with self._lock:
            self.misses += 1
            self._entries[key] = cache
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return cache, False


class CachingRunner:
    """A :class:`Runner` proxy whose ``cache_prefix`` is served from a :class:`PrefixLRU`.

    Everything else delegates to the wrapped runner, so ``decide`` runs unchanged
    and reuses a prefix across requests (and across routes).
    """

    def __init__(self, runner: Runner, prefixes: PrefixLRU):
        self.runner, self.prefixes = runner, prefixes

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runner, name)

    def cache_prefix(self, tokens: Sequence[int]) -> PrefixCache:
        return self.prefixes.get(tokens)[0]


@dataclass(frozen=True)
class ChatResult:
    text: str
    completion_tokens: int
    finish_reason: str


@dataclass
class ServerState:
    """What every route shares: the runner, its prefix LRU, and the model worker."""

    runner: Any
    prefixes: PrefixLRU
    worker: ThreadPoolExecutor
    generate: Generate | None = None
    temperature: Callable[[str, str], float | None] = field(default=lambda model, readout: None)
    log: DecisionLog | None = None

    @property
    def model_id(self) -> str:
        return getattr(self.runner, "model_id", "default")

    async def run(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the model worker thread without blocking the event loop."""
        return await asyncio.wrap_future(self.worker.submit(fn))


def mlx_generate(
    runner: Any,
    tokens: Sequence[int],
    prefix_cache: PrefixCache | None,
    max_tokens: int,
    temperature: float,
) -> ChatResult:
    """Greedy/temperature generation with ``mlx_lm.stream_generate`` from a prefix cache.

    The shared cache is deep-copied (``MLXRunner.prefill`` does the same), so
    generation never mutates the LRU's entry.
    """
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    n_cached = len(prefix_cache) if prefix_cache is not None else 0
    cache = (
        copy.deepcopy(prefix_cache.state)
        if prefix_cache is not None
        else make_prompt_cache(runner.model)
    )
    pieces, last = [], None
    for last in stream_generate(
        runner.model,
        runner.tokenizer,
        list(tokens[n_cached:]),
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=temperature),
    ):
        pieces.append(last.text)
    if last is None:
        return ChatResult("", 0, "length")
    return ChatResult("".join(pieces), last.generation_tokens, last.finish_reason or "length")


def _error_paths(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": ".".join(str(p) for p in err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in exc.errors(include_url=False)
    ]


def _unprocessable(detail: str, errors: list[dict[str, Any]] | None = None) -> Response:
    body: dict[str, Any] = {"detail": detail}
    if errors is not None:
        body["errors"] = errors
    return Response(body, status_code=422)


async def _json_body(request: Request) -> tuple[Any, Response | None]:
    try:
        return await request.json(), None
    except (SerializationException, ValueError):
        return None, _unprocessable("request body is not valid JSON")


@post("/v1/systemone", status_code=200)
async def systemone(request: Request, state: State) -> Response:
    """Typed decisions (ADR-0001 wire shape). 422 with field paths on invalid input."""
    payload, err = await _json_body(request)
    if err is not None:
        return err
    try:
        req = SystemOneRequest.model_validate(payload)
    except ValidationError as exc:
        return _unprocessable(
            "request does not match the SystemOneRequest schema", _error_paths(exc)
        )
    s: ServerState = state.server
    try:
        resp, hit = await s.run(lambda: _decide(s, req))
    except ClassifierError as exc:
        return _unprocessable(str(exc))
    return Response(
        resp.to_wire(),
        status_code=200,
        headers={"x-prefix-cache": hit},
    )


def _decide(s: ServerState, req: SystemOneRequest):
    """Run ``decide`` through the LRU; report whether its prefix was a hit."""
    opts = req.effective_options
    fitted = s.temperature(s.model_id, opts.readout)
    hits, misses = s.prefixes.hits, s.prefixes.misses
    resp = decide(req, CachingRunner(s.runner, s.prefixes), temperature=fitted)
    if s.log is not None:  # story 3.3
        applied = fitted if (opts.calibrated and fitted is not None) else 1.0
        s.log.append(decision_records(req, resp, model=s.model_id, temperature=applied))
    if s.prefixes.hits > hits:
        hit = "hit"
    elif s.prefixes.misses > misses:
        hit = "miss"
    else:
        hit = "none"
    return resp, hit


@post("/v1/chat/completions", status_code=200)
async def chat_completions(request: Request, state: State) -> Response:
    """A thin, non-streaming OpenAI-style chat completion on the same model.

    Its system turn is drawn from the shared :class:`PrefixLRU`, so a later
    decision with the same ``prefix`` reuses the KV (and vice versa).
    """
    payload, err = await _json_body(request)
    if err is not None:
        return err
    s: ServerState = state.server
    if s.generate is None:
        return Response({"detail": "this runner cannot generate"}, status_code=501)
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if (
        not isinstance(messages, list)
        or not messages
        or not all(
            isinstance(m, dict)
            and isinstance(m.get("role"), str)
            and isinstance(m.get("content"), str)
            for m in messages
        )
    ):
        return _unprocessable(
            "messages must be a non-empty list of {role, content} strings",
            [{"loc": "messages", "msg": "invalid", "type": "value_error"}],
        )
    max_tokens = int(payload.get("max_tokens") or 256)
    temp = float(payload.get("temperature") or 0.0)

    def run() -> tuple[ChatResult, int, int, str]:
        chat = s.runner.chat_template.render_messages(messages)
        tokens = list(s.runner.tokenize(chat.text))
        cache, hit = None, "none"
        if chat.prefix_text:
            prefix = s.runner.tokenize(chat.prefix_text)
            if tokens[: len(prefix)] == prefix and len(prefix) < len(tokens):
                cache, was_hit = s.prefixes.get(prefix)
                hit = "hit" if was_hit else "miss"
        out = s.generate(s.runner, tokens, cache, max_tokens, temp)
        return out, len(tokens), len(cache) if cache is not None else 0, hit

    out, n_prompt, n_cached, hit = await s.run(run)
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": s.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": out.text},
                "finish_reason": out.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": n_prompt,
            "completion_tokens": out.completion_tokens,
            "total_tokens": n_prompt + out.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": n_cached},
        },
    }
    return Response(body, status_code=200, headers={"x-prefix-cache": hit})


def create_app(
    runner: Any = None,
    *,
    load: Callable[[], Any] | None = None,
    generate: Generate | None = None,
    prefix_cache_size: int = 16,
    calibration: CalibrationConfig | None = None,
    log: str | Path | DecisionLog | None = None,
) -> Litestar:
    """The Litestar app over one ``runner``, or the one ``load()`` builds.

    Pass ``load`` for an MLX model: it runs on the model worker thread, which
    MLX requires (arrays are bound to the thread's stream). ``generate`` backs
    the chat route; it defaults to :func:`mlx_generate` when the runner is an
    ``MLXRunner`` and is otherwise off (the route answers 501). ``calibration``
    supplies the fitted temperature per (model, readout) (story 3.2); without
    one, calibrated probabilities equal raw. ``log`` appends every served
    decision to a JSONL file (story 3.3).
    """
    if (runner is None) == (load is None):
        raise ValueError("pass exactly one of runner or load")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-classifiers-model")
    if load is not None:
        runner = worker.submit(load).result()
    if generate is None and type(runner).__name__ == "MLXRunner":
        generate = mlx_generate
    server = ServerState(
        runner=runner,
        prefixes=PrefixLRU(runner.cache_prefix, prefix_cache_size),
        worker=worker,
        generate=generate,
        temperature=(calibration or CalibrationConfig()).temperature,
        log=log if isinstance(log, DecisionLog) or log is None else DecisionLog(log),
    )
    return Litestar(
        route_handlers=[systemone, chat_completions],
        state=State({"server": server}),
    )


def build_app_from_args(args: argparse.Namespace) -> Litestar:
    from .mlx_runner import MLXRunner

    cal_path = args.calibration or default_calibration_path()
    calibration = CalibrationConfig.load_or_empty(cal_path)
    print(f"calibration: {cal_path} ({'loaded' if calibration.models else 'none fitted'})")
    if args.log:
        print(f"decision log: {args.log}")
    return create_app(
        load=lambda: MLXRunner.load(args.model),
        prefix_cache_size=args.prefix_cache_size,
        calibration=calibration,
        log=args.log,
    )


def main(args: argparse.Namespace) -> int:
    import uvicorn

    app = build_app_from_args(args)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
