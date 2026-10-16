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
for the chat route's generation (non-streaming).

The chat route (story 3.5) honours ``max_tokens`` up to a server ceiling, takes
a per-call ``enable_thinking``, and splits a thinking trace out of the
generated tokens into ``message.reasoning``. Generation always runs on its own
copy of the KV: a thinking call only *peeks* the LRU (never inserts or
reorders), so a long planning call cannot evict or alter a decision prefix.

Delta prefix caching (story 3.4): decisions also reuse the longest common token
prefix with earlier prompts through the runner's ``prompt_cache``
(:class:`~mlx_classifiers.prompt_cache.PromptCacheStore`), installed by
:func:`create_app` (``delta_cache=False`` / ``serve --no-delta-cache`` turns it
off). ``x-prefix-cache-tokens`` reports the hit length of the request's first
prefill; ``x-prefix-cache`` keeps reporting the declared prefix's LRU status.
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
from .prompt_cache import DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES, PromptCacheStore
from .runner import PrefixCache, Runner
from .schema import SystemOneRequest
from .template import ThinkingMarkers

__all__ = [
    "CachingRunner",
    "ChatResult",
    "PrefixLRU",
    "ServerState",
    "build_app_from_args",
    "create_app",
    "main",
    "mlx_generate",
    "split_reasoning",
]

#: ``generate(runner, tokens, prefix_cache, max_tokens, temperature) -> ChatResult``
Generate = Callable[[Any, Sequence[int], PrefixCache | None, int, float], "ChatResult"]

#: ``max_tokens`` when a chat request gives none (story 3.5).
DEFAULT_MAX_TOKENS = 512
#: Largest ``max_tokens`` a chat request may ask for (``--max-tokens-ceiling``).
DEFAULT_MAX_TOKENS_CEILING = 8192


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

    def peek(self, tokens: Sequence[int]) -> PrefixCache | None:
        """The cached entry for ``tokens`` or None, read-only: no build, no
        insertion, no recency bump, no hit/miss count."""
        with self._lock:
            return self._entries.get(tuple(tokens))

    def get(
        self,
        tokens: Sequence[int],
        build: Callable[[Sequence[int]], PrefixCache] | None = None,
    ) -> tuple[PrefixCache, bool]:
        """``build`` overrides the constructor's builder for a miss (e.g. without
        the delta cache)."""
        key = tuple(tokens)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key], True
        cache = (build or self.build)(key)
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

    def cache_prefix(
        self, tokens: Sequence[int], *, delta: bool = True, images: Sequence[Any] = ()
    ) -> PrefixCache:
        if images:  # keyed by tokens alone, the LRU cannot tell images apart (story 6.3)
            return self.runner.cache_prefix(tokens, delta=delta, images=images)
        if delta:
            return self.prefixes.get(tokens)[0]
        return self.prefixes.get(tokens, lambda t: self.runner.cache_prefix(t, delta=False))[0]


@dataclass(frozen=True)
class ChatResult:
    """One generation. ``finish_reason`` is ``stop`` (end-of-turn token) or
    ``length`` (``max_tokens`` reached). ``token_ids`` are the generated ids
    without the stop token; when given, the route splits a thinking trace out
    of them (:func:`split_reasoning`)."""

    text: str
    completion_tokens: int
    finish_reason: str
    token_ids: tuple[int, ...] = ()


def _find(seq: Sequence[int], sub: Sequence[int], start: int = 0) -> int:
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if list(seq[i : i + n]) == list(sub):
            return i
    return -1


def split_reasoning(
    ids: Sequence[int],
    markers: ThinkingMarkers,
    tokenize: Callable[[str], list[int]],
    decode: Callable[[Sequence[int]], str],
) -> tuple[str, str, int] | None:
    """Split generated ``ids`` into ``(reasoning, content, reasoning_tokens)``.

    None when the start marker never appears. The trace is the tokens after the
    start marker up to the end marker (or to the end of generation when it is
    cut off: content is then empty), minus a leading ``markers.channel``.
    Content is everything outside the bracketed span. ``reasoning_tokens``
    counts the span including both markers. Both texts are whitespace-trimmed.
    """
    start, end = tokenize(markers.start), tokenize(markers.end)
    a = _find(ids, start)
    if not start or a < 0:
        return None
    b = _find(ids, end, a + len(start)) if end else -1
    stop = b if b >= 0 else len(ids)
    after = b + len(end) if b >= 0 else len(ids)
    reasoning = decode(ids[a + len(start) : stop])
    if markers.channel and reasoning.startswith(markers.channel):
        reasoning = reasoning[len(markers.channel) :]
    content = decode(ids[:a]) + decode(ids[after:])
    return reasoning.strip(), content.strip(), after - a


@dataclass
class ServerState:
    """What every route shares: the runner, its prefix LRU, and the model worker."""

    runner: Any
    prefixes: PrefixLRU
    worker: ThreadPoolExecutor
    generate: Generate | None = None
    temperature: Callable[[str, str], float | None] = field(default=lambda model, readout: None)
    log: DecisionLog | None = None
    max_tokens_ceiling: int = DEFAULT_MAX_TOKENS_CEILING

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
    generation runs on its own KV and never mutates the LRU's entry. Stops on
    the tokenizer's end-of-turn ids or after ``max_tokens`` generated tokens.
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
    pieces, ids, last = [], [], None
    for last in stream_generate(
        runner.model,
        runner.tokenizer,
        list(tokens[n_cached:]),
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=temperature),
    ):
        pieces.append(last.text)
        ids.append(int(last.token))
    if last is None:
        return ChatResult("", 0, "length")
    finish = last.finish_reason or "length"
    if finish == "stop":
        ids = ids[:-1]  # the end-of-turn token, which carries no text
    return ChatResult("".join(pieces), last.generation_tokens, finish, tuple(ids))


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
        resp, hit, delta_hit = await s.run(lambda: _decide(s, req))
    except ClassifierError as exc:
        return _unprocessable(str(exc))
    return Response(
        resp.to_wire(),
        status_code=200,
        headers={"x-prefix-cache": hit, "x-prefix-cache-tokens": str(delta_hit)},
    )


def _decide(s: ServerState, req: SystemOneRequest):
    """Run ``decide`` through the LRU; report whether its prefix was a hit, and
    the cached tokens of its first prefill (declared prefix or delta hit)."""
    opts = req.effective_options
    fitted = s.temperature(s.model_id, opts.readout)
    hits, misses = s.prefixes.hits, s.prefixes.misses
    calls = getattr(s.runner, "prefill_calls", None)
    n_calls = len(calls) if calls is not None else 0
    resp = decide(req, CachingRunner(s.runner, s.prefixes), temperature=fitted)
    delta_hit = calls[n_calls][0] if calls is not None and len(calls) > n_calls else 0
    if calls is not None:
        del calls[:]  # a server never reads older calls; keep the list from growing
    if s.log is not None:  # story 3.3
        applied = fitted if (opts.calibrated and fitted is not None) else 1.0
        s.log.append(decision_records(req, resp, model=s.model_id, temperature=applied))
    if s.prefixes.hits > hits:
        hit = "hit"
    elif s.prefixes.misses > misses:
        hit = "miss"
    else:
        hit = "none"
    return resp, hit, delta_hit


@post("/v1/chat/completions", status_code=200)
async def chat_completions(request: Request, state: State) -> Response:
    """A non-streaming OpenAI-style chat completion on the same model.

    Its system turn is drawn from the shared :class:`PrefixLRU`, so a later
    decision with the same ``prefix`` reuses the KV (and vice versa). With
    ``enable_thinking`` the prompt renders in thinking mode and the LRU is only
    peeked, never written. ``max_tokens`` (default 512) above the server
    ceiling answers 422 rather than being capped.
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
    max_tokens = _max_tokens(payload, s.max_tokens_ceiling)
    if isinstance(max_tokens, Response):
        return max_tokens
    thinking = payload.get("enable_thinking", False)
    if not isinstance(thinking, bool):
        return _unprocessable(
            "enable_thinking must be a boolean",
            [{"loc": "enable_thinking", "msg": "not a boolean", "type": "bool_type"}],
        )
    temp = float(payload.get("temperature") or 0.0)

    def run() -> tuple[ChatResult, int, int, str]:
        template = s.runner.chat_template
        chat = template.render_messages(messages, enable_thinking=thinking)
        tokens = list(s.runner.tokenize(chat.text))
        cache, hit = None, "none"
        if chat.prefix_text:
            prefix = s.runner.tokenize(chat.prefix_text)
            if tokens[: len(prefix)] == prefix and len(prefix) < len(tokens):
                if thinking:  # read-only: a planning call never touches the LRU
                    cache = s.prefixes.peek(prefix)
                    hit = "hit" if cache is not None else "miss"
                else:
                    cache, was_hit = s.prefixes.get(prefix)
                    hit = "hit" if was_hit else "miss"
        out = s.generate(s.runner, tokens, cache, max_tokens, temp)
        return out, len(tokens), len(cache) if cache is not None else 0, hit

    out, n_prompt, n_cached, hit = await s.run(run)
    message: dict[str, Any] = {"role": "assistant", "content": out.text}
    n_reasoning = 0
    markers = getattr(s.runner.chat_template, "thinking", None)
    decode = getattr(getattr(s.runner, "tokenizer", None), "decode", None)
    if out.token_ids and markers is not None and decode is not None:
        split = split_reasoning(out.token_ids, markers, s.runner.tokenize, decode)
        if split is not None:
            message["reasoning"], message["content"], n_reasoning = split
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": s.model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": out.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": n_prompt,
            "completion_tokens": out.completion_tokens,
            "total_tokens": n_prompt + out.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": n_cached},
            "completion_tokens_details": {"reasoning_tokens": n_reasoning},
        },
    }
    return Response(body, status_code=200, headers={"x-prefix-cache": hit})


def _max_tokens(payload: dict[str, Any], ceiling: int) -> int | Response:
    """The request's ``max_tokens`` (or ``max_completion_tokens``), else the default.

    Never silently capped: above ``ceiling`` is a 422 naming the ceiling."""
    key = "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
    raw = payload.get(key)
    if raw is None:
        return min(DEFAULT_MAX_TOKENS, ceiling)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        return _unprocessable(
            f"{key} must be a positive integer",
            [{"loc": key, "msg": "not a positive integer", "type": "int_type"}],
        )
    if raw > ceiling:
        return _unprocessable(
            f"{key} {raw} exceeds this server's ceiling of {ceiling} (--max-tokens-ceiling)",
            [{"loc": key, "msg": f"must be at most {ceiling}", "type": "less_than_equal"}],
        )
    return raw


def create_app(
    runner: Any = None,
    *,
    load: Callable[[], Any] | None = None,
    generate: Generate | None = None,
    prefix_cache_size: int = 16,
    calibration: CalibrationConfig | None = None,
    log: str | Path | DecisionLog | None = None,
    max_tokens_ceiling: int = DEFAULT_MAX_TOKENS_CEILING,
    delta_cache: bool = True,
    delta_cache_entries: int = DEFAULT_MAX_ENTRIES,
    delta_cache_bytes: int = DEFAULT_MAX_BYTES,
) -> Litestar:
    """The Litestar app over one ``runner``, or the one ``load()`` builds.

    Pass ``load`` for an MLX model: it runs on the model worker thread, which
    MLX requires (arrays are bound to the thread's stream). ``generate`` backs
    the chat route; it defaults to :func:`mlx_generate` when the runner is an
    ``MLXRunner`` and is otherwise off (the route answers 501). ``calibration``
    supplies the fitted temperature per (model, readout) (story 3.2); without
    one, calibrated probabilities equal raw. ``log`` appends every served
    decision to a JSONL file (story 3.3). ``max_tokens_ceiling`` bounds a chat
    request's ``max_tokens`` (story 3.5). ``delta_cache`` installs a fresh
    :class:`PromptCacheStore` (``delta_cache_entries`` entries, ``delta_cache_bytes``
    KV bytes) as the runner's ``prompt_cache``, or with False removes it, on any
    runner that has the attribute (story 3.4).
    """
    if (runner is None) == (load is None):
        raise ValueError("pass exactly one of runner or load")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-classifiers-model")
    if load is not None:
        runner = worker.submit(load).result()
    if hasattr(runner, "prompt_cache"):
        runner.prompt_cache = (
            PromptCacheStore(delta_cache_entries, delta_cache_bytes) if delta_cache else None
        )
    if generate is None and type(runner).__name__ == "MLXRunner":
        generate = mlx_generate
    server = ServerState(
        runner=runner,
        prefixes=PrefixLRU(runner.cache_prefix, prefix_cache_size),
        worker=worker,
        generate=generate,
        temperature=(calibration or CalibrationConfig()).temperature,
        log=log if isinstance(log, DecisionLog) or log is None else DecisionLog(log),
        max_tokens_ceiling=max_tokens_ceiling,
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
        load=lambda: MLXRunner.load(
            args.model, vision_soft_tokens=getattr(args, "vision_tokens", None)
        ),
        prefix_cache_size=args.prefix_cache_size,
        calibration=calibration,
        log=args.log,
        max_tokens_ceiling=getattr(args, "max_tokens_ceiling", DEFAULT_MAX_TOKENS_CEILING),
        delta_cache=not getattr(args, "no_delta_cache", False),
        delta_cache_entries=getattr(args, "delta_cache_entries", DEFAULT_MAX_ENTRIES),
        delta_cache_bytes=int(getattr(args, "delta_cache_mb", DEFAULT_MAX_BYTES >> 20)) << 20,
    )


def main(args: argparse.Namespace) -> int:
    import uvicorn

    app = build_app_from_args(args)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
