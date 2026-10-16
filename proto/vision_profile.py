"""Story 6.6: where does a frame encode go? Per-stage ms, slow (vendored) vs fast path.

Run on a FREE GPU (it loads the model):

    python proto/vision_profile.py                                   # E2B, LM resident
    python proto/vision_profile.py --model mlx-community/gemma-4-26B-A4B-it-qat-4bit --int8
    python proto/vision_profile.py --tower-only                      # no LM: no memory pressure
    python proto/vision_profile.py --no-compile                      # fast path, uncompiled
    python proto/vision_profile.py --dtype float32                   # tower weights cast
    python proto/vision_profile.py --soft-tokens 140                 # smaller budget (70/140/280)

Each stage is followed by ``mx.eval`` + ``mx.synchronize`` (so staged totals are a
little above the fused end-to-end number, which is printed too). ``--frames`` distinct
gridworld-shaped frames (160 x 192), the first one a warm-up (kernel JIT, compile
trace) and excluded. Prints median / min ms per stage, the achieved TFLOP/s of the
tower against a matmul FLOP count, two micro-benchmarks (one-hot vs gather position
embedding; SDPA with the all-zero mask vs none), and fast-vs-slow parity (max |d|,
bit-equal) per frame: the gate for flipping ``vision_fast`` on.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_classifiers.images import write_png
from mlx_classifiers.vision.processor import preprocess_png

E2B = "mlx-community/gemma-4-E2B-it-qat-4bit"


def frames(n: int) -> list[bytes]:
    """``n`` distinct 160 x 192 frames (a coloured square on grey, moving)."""
    out = []
    for i in range(n):
        x0, colour = 16 + 8 * i, (40 + 20 * i % 200, 160, 40)
        rows = [
            [colour if x0 <= x < x0 + 48 and 48 <= y < 112 else (90, 90, 90) for x in range(160)]
            for y in range(192)
        ]
        out.append(write_png(160, 192, rows))
    return out


def fence(*xs) -> None:
    mx.eval(*xs)
    mx.synchronize()


class Clock:
    def __init__(self):
        self.ms: dict[str, list[float]] = {}

    def run(self, name: str, fn, record: bool = True):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, mx.array) or (isinstance(out, tuple) and out):
            fence(*(out if isinstance(out, tuple) else (out,)))
        else:
            mx.synchronize()
        if record:
            self.ms.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)
        return out

    def report(self, title: str, flops: float | None = None, tower_keys=()) -> None:
        print(f"\n== {title}")
        for name, v in self.ms.items():
            print(f"  {name:<26} median {statistics.median(v):8.2f} ms   min {min(v):8.2f} ms")
        if flops and tower_keys:
            t = sum(statistics.median(self.ms[k]) for k in tower_keys if k in self.ms)
            if t:
                print(
                    f"  tower ~{flops / 1e12:.2f} TFLOP -> {flops / (t / 1e3) / 1e12:.1f} TFLOP/s"
                )


def tower_flops(vc, num_patches: int) -> float:
    """Matmul FLOPs of the encoder layers (projections, MLP, attention scores/values)."""
    d, i, h, hd = vc.hidden_size, vc.intermediate_size, vc.num_attention_heads, vc.head_dim
    hd_sdpa = next((t for t in (64, 80, 128) if hd <= t), hd)
    per_tok = 2 * (4 * d * h * hd + 3 * d * i)
    attn = 4 * num_patches**2 * h * hd_sdpa
    return vc.num_hidden_layers * (per_tok * num_patches + attn)


# -- slow (vendored) path, split into its stages ------------------------------------
def slow_frame(tower, png: bytes, clock: Clock, record: bool):
    vm = tower.vision_tower
    vc = tower.vision_config
    pix = clock.run(
        "preprocess (PIL+numpy)",
        lambda: preprocess_png(png, vc.patch_size, tower.max_soft_tokens, vc.pooling_kernel_size),
        record,
    )
    pv = clock.run("upload", lambda: mx.array(pix), record)
    _, _, H, W = pv.shape
    num_real = (H // vm.patch_size) * (W // vm.patch_size)
    out_len = num_real // vm.pooling_kernel_size**2

    def positions():
        pos, pad, _ = vm._patch_positions_single(H, W, max_patches=num_real)
        return mx.array(pos[None]), mx.array(pad[None])

    pos, pad = clock.run("patch positions", positions, record)
    emb = clock.run("patch+pos embed", lambda: vm.patch_embedder(pv, pos, pad), record)

    def mask():
        valid = ~pad
        m = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
        m = mx.where(m, mx.array(0.0, dtype=emb.dtype), mx.array(-1e4, dtype=emb.dtype))
        return mx.expand_dims(m, 1)

    m = clock.run("mask", mask, record)
    hs = clock.run("tower layers", lambda: vm.encoder(emb, pos, m), record)

    def pool():
        pooled, pmask = vm.pooler(hs, pos, pad, output_length=out_len)
        valid = pmask if pmask.shape[1] == out_len else ~pmask
        n = int(valid[0].astype(mx.int32).sum().item())  # the host sync
        out = pooled[0, :n][None]
        if vm.config.standardize:
            out = (out - vm.std_bias) * vm.std_scale
        return out

    pooled = clock.run("pool (+item sync)", pool, record)
    return clock.run("projector", lambda: tower.embed_vision(pooled)[0], record)


def fast_frame(enc, png: bytes, clock: Clock, record: bool):
    u8 = clock.run("preprocess (PIL uint8)", lambda: enc.uint8(png), record)
    img = clock.run("upload", lambda: mx.array(u8), record)
    sh = clock.run("shape tables (cached)", lambda: enc.shape(*img.shape[:2]).pos_emb, record)
    sh = enc.shape(*img.shape[:2])
    h = clock.run("patch+pos embed", lambda: enc.embed(img, sh), record)
    h = clock.run("tower layers", lambda: enc.layers(h, sh), record)
    h = clock.run("pool", lambda: enc.pool(h, sh), record)
    return clock.run("projector", lambda: enc.project(h)[0], record)


def merge_stage(runner, feats: mx.array, clock: Clock, record: bool) -> None:
    from mlx_classifiers.vision.merge import block_ids, merged_embeddings, per_layer_token_ids

    n = feats.shape[0]
    ids = mx.array([2] + [100] * 60 + [runner.image_tokens.image_token_id] * n + [100] * 60)
    pieces = [(61, n, feats)]
    bb = runner._backbone
    clock.run("merge: embeddings", lambda: merged_embeddings(bb, ids, pieces), record)
    clock.run("merge: per-layer ids", lambda: per_layer_token_ids(ids, pieces), record)
    if runner.vision_bidirectional:
        clock.run("merge: block ids", lambda: block_ids(ids.shape[0], pieces), record)


def micro(tower, H: int, W: int, reps: int = 10) -> None:
    """One-hot vs gather position embedding; one layer's SDPA with zero mask vs none."""
    from mlx_classifiers.vision.fast import _Shape

    vm = tower.vision_tower
    num = (H // 16) * (W // 16)
    pos, pad, _ = vm._patch_positions_single(H, W, max_patches=num)
    pos, pad = mx.array(pos[None]), mx.array(pad[None])

    def best(fn):
        fence(fn())
        b = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            fence(fn())
            b = min(b, time.perf_counter() - t0)
        return b * 1e3

    onehot = best(lambda: vm.patch_embedder._position_embeddings(pos, pad))
    sh = _Shape(tower.fast_encoder(), H, W)
    table = vm.patch_embedder.position_embedding_table
    gather = best(lambda: table[0][pos[0, :, 0]] + table[1][pos[0, :, 1]])
    print(
        f"\n== micro: position embedding  one-hot matmul {onehot:7.2f} ms   gather {gather:6.2f} ms"
    )
    print(
        f"   bit-equal: {mx.array_equal(sh.pos_emb, vm.patch_embedder._position_embeddings(pos, pad)).item()}"
    )

    a = vm.encoder.layers[0].self_attn
    hd = next((t for t in (64, 80, 128) if a.head_dim <= t), a.head_dim)
    dt = table.dtype
    q = mx.random.normal((1, a.num_heads, num, hd)).astype(dt)
    zero = mx.zeros((1, 1, num, num), dtype=dt)
    masked = best(lambda: mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, mask=zero))
    none = best(lambda: mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0))
    same = mx.array_equal(
        mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, mask=zero),
        mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0),
    ).item()
    n_layers = len(vm.encoder.layers)
    print(
        f"== micro: SDPA one layer  zero mask {masked:6.2f} ms   none {none:6.2f} ms"
        f"   (x{n_layers} layers)   bit-equal: {same}"
    )


def cast_tower(tower, dtype) -> None:
    """Cast the non-quantized vision tower weights (not the 4-bit projector)."""
    params = [
        (k, v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v)
        for k, v in tree_flatten(tower.vision_tower.parameters())
    ]
    tower.vision_tower.update(tree_unflatten(params))
    mx.eval(tower.parameters())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=E2B)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--path", choices=("both", "slow", "fast"), default="both")
    ap.add_argument("--no-compile", action="store_true", help="fast path uncompiled")
    ap.add_argument("--dtype", choices=("checkpoint", "bfloat16", "float16", "float32"))
    ap.add_argument("--soft-tokens", type=int, default=None, help="max_soft_tokens override")
    ap.add_argument("--tower-only", action="store_true", help="load only the vision weights")
    ap.add_argument("--int8", action="store_true", help="LM with the INT8 patches (as the bench)")
    args = ap.parse_args()

    from mlx_lm.utils import _download

    from mlx_classifiers.vision import load_vision_tower

    runner = None
    t0 = time.perf_counter()
    if args.tower_only:
        tower = load_vision_tower(_download(args.model))
    else:
        from mlx_classifiers.mlx_runner import MLXRunner

        runner = MLXRunner.load(args.model, int8=args.int8, vision=True)
        tower = runner.vision
    print(f"loaded {args.model} in {time.perf_counter() - t0:.1f} s (tower-only={args.tower_only})")
    if args.dtype and args.dtype != "checkpoint":
        cast_tower(tower, getattr(mx, args.dtype))
    if args.soft_tokens:
        tower.max_soft_tokens = args.soft_tokens
    vc = tower.vision_config
    print(
        f"tower: {vc.num_hidden_layers} layers, hidden {vc.hidden_size}, head_dim {vc.head_dim}, "
        f"dtype {tower.vision_tower.patch_embedder.input_proj.weight.dtype}, "
        f"clip {vc.use_clipped_linears}, standardize {vc.standardize}, "
        f"mlx {mx.__version__}, device {mx.default_device()}"
    )

    pngs = frames(args.frames + 1)
    enc = tower.fast_encoder()
    enc.compile = not args.no_compile
    u8 = enc.uint8(pngs[0])
    H, W = u8.shape[:2]
    flops = tower_flops(vc, (H // 16) * (W // 16))
    print(f"frame 160x192 -> {W}x{H}, {(H // 16) * (W // 16)} patches")

    slow_out, fast_out = [], []
    if args.path in ("both", "slow"):
        c, e2e = Clock(), Clock()
        for i, png in enumerate(pngs):
            out = slow_frame(tower, png, c, record=i > 0)
            if runner is not None:
                merge_stage(runner, out, c, record=i > 0)
            e2e.run(
                "end to end", lambda png=png: tower.encode_pixels(tower.pixel_values(png))[0], i > 0
            )
            slow_out.append(out)
        c.report("slow (vendored) path, staged", flops, ("tower layers",))
        e2e.report("slow path, one graph (encode_pixels + eval)", flops, ("end to end",))

    if args.path in ("both", "fast"):
        c, e2e = Clock(), Clock()
        enc_staged = type(enc)(tower, compile=False)
        for i, png in enumerate(pngs):
            out = fast_frame(enc_staged, png, c, record=i > 0)
            if runner is not None:
                merge_stage(runner, out, c, record=i > 0)
            res = e2e.run(
                f"end to end (compile={enc.compile})", lambda png=png: enc.encode_png(png), i > 0
            )
            fast_out.append(res)
        c.report("fast path, staged (uncompiled stages)", flops, ("tower layers",))
        e2e.report(
            "fast path, one graph (encode_png + eval)",
            flops,
            (f"end to end (compile={enc.compile})",),
        )

    if slow_out and fast_out:
        diffs = [
            float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
            for a, b in zip(slow_out, fast_out, strict=True)
        ]
        equal = [bool(mx.array_equal(a, b).item()) for a, b in zip(slow_out, fast_out, strict=True)]
        print(
            f"\n== parity fast vs slow: max |d| {max(diffs):.3g}; bit-equal {sum(equal)}/{len(equal)}"
        )
        print("   (flip vision_fast on only if bit-equal, or max |d| within the 6.4 tolerances)")

    micro(tower, H, W)
    print(f"\npeak MLX memory {mx.get_peak_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
