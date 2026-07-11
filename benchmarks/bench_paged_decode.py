"""Paged decode kernel benchmark.

该脚本比较 Prism-Infer 自实现 Triton paged decode kernel 与 PyTorch
SDPA reference。输出只使用实测数据，计时边界包含 `torch.cuda.synchronize()`。
"""

from __future__ import annotations

import argparse
import math
import sys
import statistics
import subprocess
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.ops.paged_decode import HAS_TRITON, paged_decode_attention


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ordered[index]


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)
    return ordered[index]


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=REPO_ROOT,
        ).strip()
    except Exception:
        return "unknown"


def _reference_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outputs = []
    if k_cache.dtype != q.dtype:
        k_cache = k_cache.to(q.dtype)
        v_cache = v_cache.to(q.dtype)
    block_size = k_cache.shape[1]
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    groups = num_heads // num_kv_heads
    for seq_idx in range(q.shape[0]):
        context_len = int(context_lens[seq_idx].item())
        pieces_k = []
        pieces_v = []
        remaining = context_len
        for block_id in block_tables[seq_idx].tolist():
            if remaining <= 0:
                break
            if block_id < 0:
                break
            take = min(block_size, remaining)
            pieces_k.append(k_cache[block_id, :take])
            pieces_v.append(v_cache[block_id, :take])
            remaining -= take
        if remaining != 0 or not pieces_k:
            raise RuntimeError(f"invalid reference block table, remaining={remaining}")
        keys = torch.cat(pieces_k, dim=0)
        values = torch.cat(pieces_v, dim=0)
        if groups != 1:
            keys = keys.repeat_interleave(groups, dim=1)
            values = values.repeat_interleave(groups, dim=1)
        q_i = q[seq_idx].unsqueeze(0).unsqueeze(2)
        k_i = keys.transpose(0, 1).unsqueeze(0)
        v_i = values.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            is_causal=False,
            scale=scale,
        )
        outputs.append(out.squeeze(0).squeeze(1))
    return torch.stack(outputs, dim=0)


def _make_inputs(
    *,
    batch: int,
    context_len: int,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    query_dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_blocks = (context_len + block_size - 1) // block_size
    num_blocks = batch * max_blocks
    q = torch.randn(
        batch,
        num_heads,
        head_dim,
        device="cuda",
        dtype=query_dtype,
    )
    k_source = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=query_dtype,
    )
    v_source = torch.randn_like(k_source)
    k_cache = k_source.to(cache_dtype)
    v_cache = v_source.to(cache_dtype)
    rows = []
    next_block = 0
    for _ in range(batch):
        row = list(range(next_block, next_block + max_blocks))
        rows.append(row)
        next_block += max_blocks
    block_tables = torch.tensor(rows, device="cuda", dtype=torch.int32)
    context_lens = torch.full((batch,), context_len, device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, block_tables, context_lens


def _measure(fn, warmup: int, repeat: int) -> tuple[list[float], torch.Tensor]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times = []
    last = None
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = perf_counter()
        last = fn()
        torch.cuda.synchronize()
        times.append((perf_counter() - start) * 1000.0)
    return times, last


def _report(label: str, times_ms: list[float], decode_tokens: int) -> None:
    total_s = sum(times_ms) / 1000.0
    token_s = decode_tokens * len(times_ms) / total_s if total_s > 0 else 0.0
    print(
        f"{label}: median={statistics.median(times_ms):.4f}ms "
        f"p90={_p90(times_ms):.4f}ms p99={_p99(times_ms):.4f}ms "
        f"min={min(times_ms):.4f}ms "
        f"max={max(times_ms):.4f}ms token/s={token_s:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--context-lens", default="256,1024,4096")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--cache-dtypes", default="bf16,fp8")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not HAS_TRITON:
        raise SystemExit("CUDA and Triton are required")

    query_dtype = torch.bfloat16
    cache_dtypes = []
    for name in [item.strip() for item in args.cache_dtypes.split(",") if item.strip()]:
        if name == "bf16":
            cache_dtypes.append((name, torch.bfloat16))
        elif name == "fp8" and hasattr(torch, "float8_e4m3fn"):
            cache_dtypes.append((name, torch.float8_e4m3fn))
        else:
            raise SystemExit(f"unsupported cache dtype: {name!r}")
    if not cache_dtypes:
        raise SystemExit("--cache-dtypes must contain at least one dtype")
    print(f"commit: {_commit()}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"query dtype: {query_dtype}")
    print(f"cache dtypes: {[name for name, _ in cache_dtypes]}")
    print(f"warmup: {args.warmup}, repeat: {args.repeat}")
    print(
        "shape config: "
        f"num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}, "
        f"head_dim={args.head_dim}, block_size={args.block_size}"
    )

    for cache_dtype_name, cache_dtype in cache_dtypes:
        for batch in [int(x) for x in args.batch_sizes.split(",") if x]:
            for context_len in [int(x) for x in args.context_lens.split(",") if x]:
                q, k_cache, v_cache, block_tables, context_lens = _make_inputs(
                    batch=batch,
                    context_len=context_len,
                    block_size=args.block_size,
                    num_heads=args.num_heads,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    query_dtype=query_dtype,
                    cache_dtype=cache_dtype,
                )
                scale = args.head_dim ** -0.5
                print("")
                print(
                    f"case: cache_dtype={cache_dtype_name}, batch={batch}, "
                    f"context_len={int(context_lens[0].item())}, "
                    f"q={list(q.shape)}, k_cache={list(k_cache.shape)}, "
                    f"block_tables={list(block_tables.shape)}"
                )
                kernel_times, kernel_out = _measure(
                    lambda: paged_decode_attention(
                        q,
                        k_cache,
                        v_cache,
                        block_tables,
                        context_lens,
                        scale,
                    ),
                    args.warmup,
                    args.repeat,
                )
                ref_times, ref_out = _measure(
                    lambda: _reference_decode(
                        q,
                        k_cache,
                        v_cache,
                        block_tables,
                        context_lens,
                        scale,
                    ),
                    args.warmup,
                    args.repeat,
                )
                diff = (kernel_out - ref_out).abs()
                print(
                    "kernel mean/std: "
                    f"{kernel_out.float().mean().item():.6e} / "
                    f"{kernel_out.float().std().item():.6e}"
                )
                print(
                    "reference mean/std: "
                    f"{ref_out.float().mean().item():.6e} / "
                    f"{ref_out.float().std().item():.6e}"
                )
                print(f"max diff: {diff.max().item():.6e}")
                print(f"mean diff: {diff.float().mean().item():.6e}")
                if diff.max().item() >= 1e-2:
                    print("correctness: FAIL")
                    raise RuntimeError(
                        "paged decode benchmark correctness failed: "
                        f"cache_dtype={cache_dtype_name}, batch={batch}, "
                        f"context_len={context_len}, max_diff={diff.max().item()}"
                    )
                print("correctness: PASS")
                _report("kernel", kernel_times, batch)
                _report("reference", ref_times, batch)
                print(
                    "memory: "
                    f"allocated={torch.cuda.memory_allocated() / 2**20:.2f}MiB "
                    f"reserved={torch.cuda.memory_reserved() / 2**20:.2f}MiB "
                    f"peak={torch.cuda.max_memory_allocated() / 2**20:.2f}MiB"
                )


if __name__ == "__main__":
    main()
