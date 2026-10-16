"""``mlx-classifiers memory``: what each loading stage costs (story 2.9).

Loads a model the way the server does and samples memory after each stage:

1. ``base weights``: ``mlx_lm.load`` (every parameter evaluated);
2. ``int8 patch`` (``--int8``): the W8A8 dense wrappers and, on a MoE, the W4A8
   experts (story 2.3, 2.8);
3. ``vision tower`` (``--vision``): the vendored tower's weights (story 6.3);
4. ``delta store``: ``K`` decisions over states of ``L`` tokens that differ in
   their tail, so the store holds ``K`` KV entries (story 3.4);
5. ``one decision``: one more delta-cached decision.

Each sample has the process's resident set and physical footprint (macOS
``task_info(TASK_VM_INFO)``; the footprint is what Activity Monitor calls Memory and
includes Metal buffers), the max resident set so far (``getrusage``), and MLX's
active, peak and cache bytes. ``peak`` is per stage: it is reset after every sample.
``held`` is the bytes of the arrays a stage added (model tree, vision tower, store
entries), counted directly. The table prints each value and its change from the
previous stage.
"""

from __future__ import annotations

import argparse
import ctypes
import resource
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Audit", "Sample", "process_memory", "run_audit", "tree_nbytes"]

GB = 1e9


class _TaskVMInfo(ctypes.Structure):
    # <mach/task_info.h> task_vm_info, up to phys_footprint (rev 1)
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_peak", ctypes.c_uint64),
        *((name, ctypes.c_uint64) for name in (
            "device", "device_peak", "internal", "internal_peak", "external",
            "external_peak", "reusable", "reusable_peak", "purgeable_volatile_pmap",
            "purgeable_volatile_resident", "purgeable_volatile_virtual", "compressed",
            "compressed_peak", "compressed_lifetime", "phys_footprint",
        )),
    ]  # fmt: skip


def process_memory() -> dict[str, int | None]:
    """``{"rss", "footprint", "max_rss"}`` in bytes for this process. ``footprint``
    is None off macOS; ``rss`` falls back to ``/proc/self/statm``."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # ru_maxrss is bytes on macOS, KiB on Linux
        libc = ctypes.CDLL(None)
        info = _TaskVMInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
        task = ctypes.c_uint32.in_dll(libc, "mach_task_self_")
        if libc.task_info(task, 22, ctypes.byref(info), ctypes.byref(count)) == 0:
            return {
                "rss": int(info.resident_size),
                "footprint": int(info.phys_footprint),
                "max_rss": int(max_rss),
            }
        return {"rss": None, "footprint": None, "max_rss": int(max_rss)}
    try:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * resource.getpagesize()
    except OSError:
        rss = None
    return {"rss": rss, "footprint": None, "max_rss": int(max_rss) * 1024}


def _mlx_memory() -> dict[str, int]:
    import mlx.core as mx

    out = {
        "active": mx.get_active_memory(),
        "peak": mx.get_peak_memory(),
        "cache": mx.get_cache_memory(),
    }
    mx.reset_peak_memory()
    return out


def tree_nbytes(obj: Any, seen: set[int] | None = None) -> int:
    """Bytes of every ``mx.array`` reachable from ``obj`` through dicts, lists, tuples,
    ``nn.Module`` (including underscored attributes, which ``parameters()`` skips) and
    plain objects' ``__dict__``; each array counted once."""
    import mlx.core as mx

    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, mx.array):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(tree_nbytes(v, seen) for v in obj.values())
    if isinstance(obj, list | tuple):
        return sum(tree_nbytes(v, seen) for v in obj)
    d = getattr(obj, "__dict__", None)
    if d and not isinstance(obj, type):
        return sum(tree_nbytes(v, seen) for v in d.values())
    return 0


@dataclass
class Sample:
    stage: str
    rss: int | None
    footprint: int | None
    max_rss: int
    active: int
    peak: int
    cache: int
    held: int | None = None
    note: str = ""


@dataclass
class Audit:
    """Samples in stage order. ``probe`` and ``mlx`` are injectable for tests."""

    probe: Callable[[], dict[str, int | None]] = process_memory
    mlx: Callable[[], dict[str, int]] = _mlx_memory
    samples: list[Sample] = field(default_factory=list)

    def mark(self, stage: str, held: int | None = None, note: str = "") -> Sample:
        s = Sample(stage=stage, **self.probe(), **self.mlx(), held=held, note=note)  # type: ignore[arg-type]
        self.samples.append(s)
        return s

    def render(self) -> str:
        """Markdown: footprint, rss, active and cache with their change from the
        previous stage; max rss, the stage's MLX peak and the bytes it holds, as is."""
        deltas, plain = ("footprint", "rss", "active", "cache"), ("max_rss", "peak", "held")

        def gb(v: int | None) -> str:
            return "-" if v is None else f"{v / GB:.2f}"

        head = "| stage | " + " | ".join(f"{c} GB | d{c}" for c in deltas)
        head += " | " + " | ".join(f"{c} GB" for c in plain) + " | note |"
        lines = [head, "|---|" + "---|---|" * len(deltas) + "---|" * len(plain) + "---|"]
        prev: Sample | None = None
        for s in self.samples:
            cells = []
            for c in deltas:
                v, p = getattr(s, c), None if prev is None else getattr(prev, c)
                cells += [gb(v), "-" if v is None or p is None else f"{(v - p) / GB:+.2f}"]
            cells += [gb(getattr(s, c)) for c in plain]
            lines.append(f"| {s.stage} | " + " | ".join(cells) + f" | {s.note} |")
            prev = s
        return "\n".join(lines)


def run_audit(
    model_id: str,
    *,
    int8: bool = False,
    int8_experts: bool = True,
    vision: bool = False,
    store_mb: int = 2048,
    entries: int = 4,
    tokens: int = 1024,
    tail: int = 50,
    audit: Audit | None = None,
) -> Audit:
    """Load ``model_id`` stage by stage (see the module docstring) and sample each."""
    import mlx.core as mx

    from .answers import build_answer
    from .bench import PREFIX, QUESTION, delta_states
    from .mlx_runner import MLXRunner
    from .prompt import compile_question
    from .prompt_cache import PromptCacheStore
    from .readout import bind_options
    from .schema import SystemOneRequest

    audit = audit or Audit()
    mx.clear_cache()
    audit.mark("start (imports)")

    def stage(name: str, model: Any) -> None:
        total = tree_nbytes(model)
        if name == "base weights":
            audit.mark(name, total)
        else:  # the int8 patch: what the model tree gained (or lost)
            base = audit.samples[-1].held or 0
            audit.mark(name, total - base, f"model arrays now {total / GB:.2f} GB")

    runner = MLXRunner.load(
        model_id,
        int8=int8,
        int8_experts=int8_experts,
        vision=False,
        on_stage=stage,
        prompt_cache=PromptCacheStore(max_entries=max(entries, 1), max_bytes=store_mb << 20),
    )
    if vision:
        tower = runner.load_vision()
        audit.mark("vision tower", tree_nbytes(tower))
    states = delta_states(runner.tokenize, tokens, tail, entries + 1)
    prompts = []
    for st in states:
        req = SystemOneRequest.model_validate(
            {"prefix": PREFIX, "state": st, "questions": {"q": QUESTION}}
        )
        p = compile_question(req, "q", template=runner.chat_template, tokenize=runner.tokenize)
        prompts.append((p, bind_options(p, runner.tokenize)))
    store = runner.prompt_cache
    assert store is not None

    def decide(p: Any, b: Any) -> Any:
        r = runner.prefill(p.tokens, None, [p.readout_position])
        build_answer(p, b, r.at(p.readout_position))
        return r

    for p, b in prompts[:entries]:
        decide(p, b)
    audit.mark(
        f"delta store ({entries} x {len(prompts[0][0].tokens)} tok)",
        store.nbytes,
        f"{len(store)} entries, cap {store_mb} MiB",
    )
    r = decide(*prompts[entries])
    audit.mark("one decision", None, f"hit {r.cached_tokens}, prefilled {r.input_tokens}")
    return audit


def add_arguments(p: argparse.ArgumentParser) -> None:
    from .mlx_runner import DEFAULT_MODEL

    p.add_argument("--model", default=DEFAULT_MODEL, help="mlx-lm repo or path to load")
    p.add_argument("--int8", action="store_true", help="apply the INT8 patches (M5)")
    p.add_argument(
        "--no-int8-experts", action="store_true", help="with --int8, keep stock MoE experts"
    )
    p.add_argument("--vision", action="store_true", help="load the vision tower")
    p.add_argument("--delta-store-mb", type=int, default=2048, help="delta store cap in MiB")
    p.add_argument("--store-entries", type=int, default=4, help="decisions that warm the store")
    p.add_argument("--store-tokens", type=int, default=1024, help="state tokens per decision")
    p.set_defaults(func=main)


def main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    audit = run_audit(
        args.model,
        int8=args.int8,
        int8_experts=not args.no_int8_experts,
        vision=args.vision,
        store_mb=args.delta_store_mb,
        entries=args.store_entries,
        tokens=args.store_tokens,
    )
    flags = " ".join(f for f, on in (("--int8", args.int8), ("--vision", args.vision)) if on)
    print(f"model {args.model} {flags}  mlx {mx.__version__}")
    print(audit.render())
    return 0
