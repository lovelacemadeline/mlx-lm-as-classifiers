"""Delta prefix caching: KV states keyed by token sequences (story 3.4).

A :class:`PromptCacheStore` keeps the KV state a runner had after prefilling
some token sequence, and answers ``lookup(tokens)`` with the longest prefix of
``tokens`` it can serve: the entry sharing the longest common token prefix with
``tokens``, rewound to that length. A runner then prefills only the delta.

The store knows nothing about KV layouts. Rewinding is the runner's business,
passed in as ``rewind(state, stored_len, hit_len) -> state | None``: it returns
a *new* state holding exactly the first ``hit_len`` tokens of ``state`` (never
mutating the stored one), or None when that length is not representable (a
sliding-window cache that has already dropped the rows it would need). The
store then tries the next-best entry, and finally reports no hit.

mlx-lm 0.31.3 ships ``PromptTrie`` / ``LRUPromptCache``. They are not reused:
``fetch_nearest_cache`` considers one "longer" and one "shorter" candidate and
trims with ``RotatingKVCache.trim``, which is only correct before the ring has
slid (``is_trimmable`` is ``offset < max_size``), and its insert drops every
stored prefix of a trimmable entry. With at most a few dozen entries a linear
scan with a C-speed common-prefix search is simpler and lets every entry be
tried in order of hit length.

Bounded LRU by entry count and by total bytes (``nbytes`` per entry, given by
the runner). A hit bumps recency. Thread-safe.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "PromptCacheStore",
    "Rewind",
    "common_prefix_len",
    "without_delta",
]

#: Entries kept by default (``serve --delta-cache-entries``).
DEFAULT_MAX_ENTRIES = 8
#: Total KV bytes kept by default (``serve --delta-cache-mb``): 2 GiB.
DEFAULT_MAX_BYTES = 2 << 30

#: ``rewind(state, stored_len, hit_len) -> new state | None`` (None: not representable).
Rewind = Callable[[Any, int, int], Any]


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token tuples.

    Binary search on slice equality, so the comparisons run in C."""
    a, b = tuple(a), tuple(b)
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:  # largest k with a[:k] == b[:k]
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


@dataclass
class _Entry:
    state: Any
    nbytes: int


class PromptCacheStore:
    """Bounded LRU of KV states keyed by the token sequence they cover."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, max_bytes: int = DEFAULT_MAX_BYTES):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries: OrderedDict[tuple[int, ...], _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.nbytes = 0
        self.hits = 0
        self.misses = 0
        #: Rewinds refused as not representable (sliding window already slid).
        self.unrepresentable = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, tokens: Sequence[int]) -> bool:
        return tuple(tokens) in self._entries

    def keys(self) -> list[tuple[int, ...]]:
        """Stored sequences, least recently used first."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.nbytes = 0

    def lookup(
        self, tokens: Sequence[int], rewind: Rewind, *, max_len: int | None = None, min_len: int = 0
    ) -> tuple[int, Any]:
        """``(hit_len, state)``: the longest representable cached prefix of ``tokens``.

        ``hit_len`` is at most ``max_len`` (default ``len(tokens)``); a hit no
        longer than ``min_len`` (what the caller already has, e.g. a declared
        prefix) is not worth a rewind and reports ``(0, None)``, as does no hit.
        ``state`` is the runner's fresh copy from ``rewind``, never a stored object.
        """
        tokens = tuple(tokens)
        cap = len(tokens) if max_len is None else min(max_len, len(tokens))
        with self._lock:
            candidates = [(min(common_prefix_len(k, tokens), cap), k) for k in self._entries]
            candidates.sort(key=lambda c: -c[0])
            for hit, key in candidates:
                if hit <= min_len:
                    break
                state = rewind(self._entries[key].state, len(key), hit)
                if state is None:
                    self.unrepresentable += 1
                    continue
                self._entries.move_to_end(key)
                self.hits += 1
                return hit, state
            self.misses += 1
            return 0, None

    def insert(self, tokens: Sequence[int], state: Any, nbytes: int) -> bool:
        """Store ``state`` (already the caller's own copy) as covering ``tokens``.

        An existing key only has its recency bumped. An entry larger than
        ``max_bytes`` is not stored. Evicts least recently used entries until both
        bounds hold. Returns whether ``state`` was stored."""
        key = tuple(tokens)
        if not key:
            return False
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return False
            if nbytes > self.max_bytes:
                return False
            self._entries[key] = _Entry(state, nbytes)
            self.nbytes += nbytes
            while len(self._entries) > self.max_entries or self.nbytes > self.max_bytes:
                _, old = self._entries.popitem(last=False)
                self.nbytes -= old.nbytes
            return True


class _NoDelta:
    """A runner proxy calling ``prefill`` / ``cache_prefix`` with ``delta=False``."""

    def __init__(self, runner: Any):
        self.runner = runner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runner, name)

    def cache_prefix(self, tokens: Sequence[int]) -> Any:
        return self.runner.cache_prefix(tokens, delta=False)

    def prefill(self, tokens: Sequence[int], prefix_cache=None, readout_positions=None) -> Any:
        return self.runner.prefill(tokens, prefix_cache, readout_positions, delta=False)


def without_delta(runner: Any) -> Any:
    """``runner`` with delta prefix caching off for every call made through it
    (no lookup, no insert), or ``runner`` itself when it has no store."""
    if getattr(runner, "prompt_cache", None) is None:
        return runner
    return _NoDelta(runner)
