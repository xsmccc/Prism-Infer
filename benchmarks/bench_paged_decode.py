"""Paged decode kernel correctness/performance benchmark.

该脚本比较 Prism-Infer 自实现 Triton paged decode kernel 与独立 PyTorch
SDPA reference。它支持固定随机输入的多 page-size matrix，并把每个 latency
sample、correctness、显存、Git 和 GPU 身份写入 JSON/JSONL。

计时边界使用 host ``perf_counter``，每次测量前后都执行
``torch.cuda.synchronize()``。结果是 kernel microbenchmark，不能直接表述为
full-engine TPOT、吞吐或 GPU utilization。
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import collect_git_metadata, collect_gpu_metadata
from prism_infer.analysis.benchmark_schema import summarize_values
from prism_infer.ops.paged_decode import HAS_TRITON, paged_decode_attention


PAGED_DECODE_BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_CONTEXT_LENS = (256, 1024, 4096)
DEFAULT_PAGE_SIZES = (256,)
DEFAULT_NUM_QUERY_HEADS = 32
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_CACHE_DTYPES = ("bf16", "fp8")
DEFAULT_WARMUP = 10
DEFAULT_REPEAT = 50
DEFAULT_SEED = 20260717
DEFAULT_MAX_ABS_DIFF = 1e-2
DEFAULT_MEAN_ABS_DIFF = 1e-3
MIB = 2**20


@dataclass(frozen=True)
class Measurement:
    """一次实现的所有 timing samples、最后输出和 allocator 证据。"""

    samples_ms: tuple[float, ...]
    output: torch.Tensor
    memory_bytes: dict[str, int]


def _csv(values: tuple[object, ...]) -> str:
    return ",".join(str(value) for value in values)


def _parse_positive_int_csv(raw: str, *, option_name: str) -> tuple[int, ...]:
    """解析非空、无重复的正整数 CSV 参数。"""

    pieces = [piece.strip() for piece in raw.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError(f"{option_name} must be a non-empty integer CSV, got {raw!r}")
    try:
        values = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"{option_name} must contain integers, got {raw!r}") from exc
    if any(value <= 0 for value in values):
        raise ValueError(f"{option_name} values must be positive, got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"{option_name} values must be unique, got {values}")
    return values


def _framework_metadata() -> dict[str, str]:
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        try:
            import triton

            triton_version = str(getattr(triton, "__version__", "unknown"))
        except ImportError:
            triton_version = "unknown"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "triton": triton_version,
    }


def _validate_preflight(
    gpu: dict[str, Any],
    *,
    max_memory_used_mib: float | None,
    max_utilization_percent: float | None,
) -> dict[str, Any]:
    """执行可选 GPU 空闲门禁，并返回写入结果的审计记录。"""

    snapshot = gpu["nvidia_smi"]
    thresholds = {
        "max_memory_used_mib": max_memory_used_mib,
        "max_utilization_gpu_percent": max_utilization_percent,
    }
    if max_memory_used_mib is None and max_utilization_percent is None:
        return {"enabled": False, "passed": None, "thresholds": thresholds}
    if not snapshot.get("available"):
        raise RuntimeError("GPU preflight requested, but nvidia-smi UUID lookup failed")

    failures = []
    observed_memory = snapshot.get("memory_used_mib")
    observed_utilization = snapshot.get("utilization_gpu_percent")
    if max_memory_used_mib is not None:
        if observed_memory is None or observed_memory > max_memory_used_mib:
            failures.append(f"memory.used={observed_memory} MiB > {max_memory_used_mib} MiB")
    if max_utilization_percent is not None:
        if observed_utilization is None or observed_utilization > max_utilization_percent:
            failures.append(f"utilization.gpu={observed_utilization}% > {max_utilization_percent}%")
    if failures:
        raise RuntimeError("GPU preflight failed: " + "; ".join(failures))
    return {"enabled": True, "passed": True, "thresholds": thresholds}


def _reference_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """收集 paged K/V 后调用 PyTorch SDPA 的独立 reference。"""

    outputs = []
    if k_cache.dtype != q.dtype:
        k_cache = k_cache.to(q.dtype)
        v_cache = v_cache.to(q.dtype)
    page_size = k_cache.shape[1]
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    groups = num_query_heads // num_kv_heads
    for sequence_index in range(q.shape[0]):
        context_len = int(context_lens[sequence_index].item())
        pieces_k = []
        pieces_v = []
        remaining = context_len
        for block_id in block_tables[sequence_index].tolist():
            if remaining <= 0:
                break
            if block_id < 0:
                break
            take = min(page_size, remaining)
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
        # q_i: [1, query_heads, 1, head_dim]
        # k_i/v_i: [1, query_heads, context_len, head_dim]
        q_i = q[sequence_index].unsqueeze(0).unsqueeze(2)
        k_i = keys.transpose(0, 1).unsqueeze(0)
        v_i = values.transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            is_causal=False,
            scale=scale,
        )
        outputs.append(output.squeeze(0).squeeze(1))
    return torch.stack(outputs, dim=0)


def _make_inputs(
    *,
    batch: int,
    context_len: int,
    page_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    query_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 page-independent deterministic logical Q/K/V，再打包到物理页。"""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        batch,
        num_query_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=query_dtype,
    )
    # dense K/V: [batch, context_len, kv_heads, head_dim]
    dense_shape = (batch, context_len, num_kv_heads, head_dim)
    dense_k = torch.randn(
        dense_shape,
        generator=generator,
        device=device,
        dtype=query_dtype,
    )
    dense_v = torch.randn(
        dense_shape,
        generator=generator,
        device=device,
        dtype=query_dtype,
    )

    pages_per_sequence = (context_len + page_size - 1) // page_size
    padded_context_len = pages_per_sequence * page_size
    padded_shape = (batch, padded_context_len, num_kv_heads, head_dim)
    padded_k = torch.zeros(padded_shape, device=device, dtype=query_dtype)
    padded_v = torch.zeros_like(padded_k)
    padded_k[:, :context_len].copy_(dense_k)
    padded_v[:, :context_len].copy_(dense_v)
    cache_shape = (
        batch * pages_per_sequence,
        page_size,
        num_kv_heads,
        head_dim,
    )
    k_cache = padded_k.view(cache_shape).to(cache_dtype)
    v_cache = padded_v.view(cache_shape).to(cache_dtype)
    block_tables = torch.arange(
        batch * pages_per_sequence,
        device=device,
        dtype=torch.int32,
    ).view(batch, pages_per_sequence)
    context_lens = torch.full(
        (batch,),
        context_len,
        device=device,
        dtype=torch.int32,
    )
    return q, k_cache, v_cache, block_tables, context_lens


def _memory_snapshot(device: torch.device, allocated_before: int) -> dict[str, int]:
    allocated_after = torch.cuda.memory_allocated(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    return {
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "reserved_after": torch.cuda.memory_reserved(device),
        "peak_allocated": peak_allocated,
        "peak_delta": max(0, peak_allocated - allocated_before),
    }


def _measure(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> Measurement:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    samples_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize(device)
        started = perf_counter()
        output = fn()
        torch.cuda.synchronize(device)
        samples_ms.append((perf_counter() - started) * 1000.0)
    if output is None:
        raise RuntimeError("measurement produced no output")
    return Measurement(
        samples_ms=tuple(samples_ms),
        output=output,
        memory_bytes=_memory_snapshot(device, allocated_before),
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _latency_record(samples_ms: tuple[float, ...], decode_tokens: int) -> dict[str, Any]:
    token_rates = [decode_tokens * 1000.0 / sample for sample in samples_ms]
    return {
        "samples_ms": list(samples_ms),
        "stats_ms": summarize_values(samples_ms),
        "tokens_per_second": summarize_values(token_rates),
    }


def _output_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.float()
    return {
        "mean": values.mean().item(),
        "std": values.std().item(),
    }


def _build_case_record(
    *,
    cache_dtype_name: str,
    batch: int,
    context_len: int,
    page_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
    scale: float,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    kernel: Measurement,
    reference: Measurement,
    max_abs_diff_limit: float,
    mean_abs_diff_limit: float,
) -> dict[str, Any]:
    """把一个 matrix cell 变成自包含、可 JSON 序列化的记录。"""

    difference = (kernel.output - reference.output).abs()
    max_abs_diff = difference.max().item()
    mean_abs_diff = difference.float().mean().item()
    correctness_passed = max_abs_diff < max_abs_diff_limit and mean_abs_diff < mean_abs_diff_limit
    kernel_latency = _latency_record(kernel.samples_ms, batch)
    reference_latency = _latency_record(reference.samples_ms, batch)
    kernel_median = float(kernel_latency["stats_ms"]["median"])
    reference_median = float(reference_latency["stats_ms"]["median"])
    tensor_memory = {
        "q": _tensor_bytes(q),
        "k_cache": _tensor_bytes(k_cache),
        "v_cache": _tensor_bytes(v_cache),
        "block_tables": _tensor_bytes(block_tables),
        "context_lens": _tensor_bytes(context_lens),
    }
    tensor_memory["total"] = sum(tensor_memory.values())
    logical_cache_bytes = batch * context_len * num_kv_heads * head_dim * k_cache.element_size() * 2
    return {
        "case_id": (f"{cache_dtype_name}_page{page_size}_batch{batch}_context{context_len}"),
        "parameters": {
            "cache_dtype": cache_dtype_name,
            "query_dtype": str(q.dtype),
            "batch": batch,
            "context_len": context_len,
            "page_size": page_size,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "gqa_group_size": num_query_heads // num_kv_heads,
            "seed": seed,
            "attention_scale": scale,
        },
        "shapes": {
            "q": list(q.shape),
            "k_cache": list(k_cache.shape),
            "v_cache": list(v_cache.shape),
            "block_tables": list(block_tables.shape),
            "context_lens": list(context_lens.shape),
            "output": list(kernel.output.shape),
        },
        "correctness": {
            "passed": correctness_passed,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "limits": {
                "max_abs_diff_exclusive": max_abs_diff_limit,
                "mean_abs_diff_exclusive": mean_abs_diff_limit,
            },
            "kernel_output": _output_stats(kernel.output),
            "reference_output": _output_stats(reference.output),
        },
        "latency": {
            "kernel": kernel_latency,
            "reference": reference_latency,
            "reference_over_kernel_median": reference_median / kernel_median,
        },
        "memory_bytes": {
            "input_tensors": tensor_memory,
            "logical_kv_payload": logical_cache_bytes,
            "physical_kv_payload": tensor_memory["k_cache"] + tensor_memory["v_cache"],
            "kernel": kernel.memory_bytes,
            "reference": reference.memory_bytes,
        },
    }


def _print_case(record: dict[str, Any]) -> None:
    parameters = record["parameters"]
    correctness = record["correctness"]
    kernel = record["latency"]["kernel"]
    reference = record["latency"]["reference"]
    print("")
    print(
        "case: "
        f"cache_dtype={parameters['cache_dtype']}, "
        f"page_size={parameters['page_size']}, batch={parameters['batch']}, "
        f"context_len={parameters['context_len']}"
    )
    print(
        "shapes: "
        f"q={record['shapes']['q']}, k_cache={record['shapes']['k_cache']}, "
        f"block_tables={record['shapes']['block_tables']}"
    )
    print(
        "correctness: "
        f"{'PASS' if correctness['passed'] else 'FAIL'} "
        f"max_diff={correctness['max_abs_diff']:.6e} "
        f"mean_diff={correctness['mean_abs_diff']:.6e}"
    )
    for label, latency in (("kernel", kernel), ("reference", reference)):
        stats = latency["stats_ms"]
        rates = latency["tokens_per_second"]
        print(
            f"{label}: median={stats['median']:.4f}ms "
            f"p90={stats['p90']:.4f}ms p99={stats['p99']:.4f}ms "
            f"min={stats['min']:.4f}ms max={stats['max']:.4f}ms "
            f"median_token/s={rates['median']:.2f}"
        )
    memory = record["memory_bytes"]
    print(
        "memory: "
        f"physical_kv={memory['physical_kv_payload'] / MIB:.2f}MiB "
        f"kernel_peak={memory['kernel']['peak_allocated'] / MIB:.2f}MiB "
        f"reference_peak={memory['reference']['peak_allocated'] / MIB:.2f}MiB"
    )


def _resolve_output_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "json" if path.suffix.lower() == ".json" else "jsonl"


def _write_output(
    path: Path,
    *,
    output_format: str,
    overwrite: bool,
    run: dict[str, Any],
    cases: list[dict[str, Any]],
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        document = {
            "schema_version": PAGED_DECODE_BENCHMARK_SCHEMA_VERSION,
            "record_type": "prism_paged_decode_benchmark_run",
            "run": run,
            "cases": cases,
        }
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            record = {
                "schema_version": PAGED_DECODE_BENCHMARK_SCHEMA_VERSION,
                "record_type": "prism_paged_decode_benchmark_case",
                "run": run,
                "case": case,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_cache_dtypes(raw: str) -> tuple[tuple[str, torch.dtype], ...]:
    names = tuple(piece.strip() for piece in raw.split(","))
    if not names or any(not name for name in names):
        raise ValueError("--cache-dtypes must be a non-empty CSV")
    if len(set(names)) != len(names):
        raise ValueError(f"--cache-dtypes values must be unique, got {names}")
    resolved = []
    for name in names:
        if name == "bf16":
            resolved.append((name, torch.bfloat16))
        elif name == "fp8" and hasattr(torch, "float8_e4m3fn"):
            resolved.append((name, torch.float8_e4m3fn))
        else:
            raise ValueError(f"unsupported cache dtype: {name!r}")
    return tuple(resolved)


def _reset_cuda_allocator(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default=_csv(DEFAULT_BATCH_SIZES))
    parser.add_argument("--context-lens", default=_csv(DEFAULT_CONTEXT_LENS))
    page_group = parser.add_mutually_exclusive_group()
    page_group.add_argument(
        "--page-sizes",
        help="physical KV page-size CSV, for example 16,32,64,128,256",
    )
    page_group.add_argument(
        "--block-size",
        type=int,
        help="backwards-compatible single page-size alias",
    )
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_QUERY_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--cache-dtypes", default=_csv(DEFAULT_CACHE_DTYPES))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-abs-diff", type=float, default=DEFAULT_MAX_ABS_DIFF)
    parser.add_argument("--mean-abs-diff", type=float, default=DEFAULT_MEAN_ABS_DIFF)
    parser.add_argument("--max-start-memory-used-mib", type=float)
    parser.add_argument("--max-start-gpu-utilization", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-format",
        choices=("auto", "json", "jsonl"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    if not torch.cuda.is_available() or not HAS_TRITON:
        raise SystemExit("CUDA and Triton are required")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        parser.error(f"--device must be in [0, {torch.cuda.device_count() - 1}], got {args.device}")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be non-negative and --repeat must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.num_heads <= 0 or args.num_kv_heads <= 0 or args.head_dim <= 0:
        parser.error("head counts and --head-dim must be positive")
    if args.num_heads % args.num_kv_heads != 0:
        parser.error("--num-heads must be divisible by --num-kv-heads")
    if args.max_abs_diff <= 0 or args.mean_abs_diff <= 0:
        parser.error("correctness thresholds must be positive")

    try:
        batch_sizes = _parse_positive_int_csv(args.batch_sizes, option_name="--batch-sizes")
        context_lens = _parse_positive_int_csv(args.context_lens, option_name="--context-lens")
        if args.page_sizes is not None:
            page_sizes = _parse_positive_int_csv(args.page_sizes, option_name="--page-sizes")
        elif args.block_size is not None:
            if args.block_size <= 0:
                raise ValueError("--block-size must be positive")
            page_sizes = (args.block_size,)
        else:
            page_sizes = DEFAULT_PAGE_SIZES
        cache_dtypes = _resolve_cache_dtypes(args.cache_dtypes)
    except ValueError as exc:
        parser.error(str(exc))

    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"output already exists; pass --overwrite: {args.output}")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gpu = collect_gpu_metadata(args.device).detailed_dict()
    try:
        preflight = _validate_preflight(
            gpu,
            max_memory_used_mib=args.max_start_memory_used_mib,
            max_utilization_percent=args.max_start_gpu_utilization,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    run = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "status": "running",
        "git": collect_git_metadata(REPO_ROOT).as_dict(),
        "framework": _framework_metadata(),
        "gpu": gpu,
        "preflight": preflight,
        "measurement": {
            "timing_method": "host_perf_counter_full_cuda_synchronize",
            "cuda_synchronize_before_and_after_each_sample": True,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "matrix": {
            "batch_sizes": list(batch_sizes),
            "context_lens": list(context_lens),
            "page_sizes": list(page_sizes),
            "cache_dtypes": [name for name, _ in cache_dtypes],
        },
        "command": list(sys.argv),
    }

    print(f"commit: {run['git']['commit']}")
    print(f"git dirty: {run['git']['dirty']}")
    print(f"gpu: {gpu['name']} ({gpu['gpu_uuid']})")
    print(f"torch/cuda/triton: {run['framework']}")
    print(f"seed: {args.seed}, warmup: {args.warmup}, repeat: {args.repeat}")
    print(
        "matrix: "
        f"page_sizes={list(page_sizes)}, batch_sizes={list(batch_sizes)}, "
        f"context_lens={list(context_lens)}, "
        f"cache_dtypes={[name for name, _ in cache_dtypes]}"
    )

    cases: list[dict[str, Any]] = []
    query_dtype = torch.bfloat16
    with torch.inference_mode():
        for cache_dtype_name, cache_dtype in cache_dtypes:
            for page_size in page_sizes:
                for batch in batch_sizes:
                    for context_len in context_lens:
                        _reset_cuda_allocator(device)
                        q, k_cache, v_cache, block_tables, case_context_lens = _make_inputs(
                            batch=batch,
                            context_len=context_len,
                            page_size=page_size,
                            num_query_heads=args.num_heads,
                            num_kv_heads=args.num_kv_heads,
                            head_dim=args.head_dim,
                            query_dtype=query_dtype,
                            cache_dtype=cache_dtype,
                            device=device,
                            seed=args.seed,
                        )
                        scale = args.head_dim**-0.5
                        kernel = _measure(
                            lambda: paged_decode_attention(
                                q,
                                k_cache,
                                v_cache,
                                block_tables,
                                case_context_lens,
                                scale,
                            ),
                            warmup=args.warmup,
                            repeat=args.repeat,
                            device=device,
                        )
                        reference = _measure(
                            lambda: _reference_decode(
                                q,
                                k_cache,
                                v_cache,
                                block_tables,
                                case_context_lens,
                                scale,
                            ),
                            warmup=args.warmup,
                            repeat=args.repeat,
                            device=device,
                        )
                        record = _build_case_record(
                            cache_dtype_name=cache_dtype_name,
                            batch=batch,
                            context_len=context_len,
                            page_size=page_size,
                            num_query_heads=args.num_heads,
                            num_kv_heads=args.num_kv_heads,
                            head_dim=args.head_dim,
                            seed=args.seed,
                            scale=scale,
                            q=q,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            block_tables=block_tables,
                            context_lens=case_context_lens,
                            kernel=kernel,
                            reference=reference,
                            max_abs_diff_limit=args.max_abs_diff,
                            mean_abs_diff_limit=args.mean_abs_diff,
                        )
                        cases.append(record)
                        _print_case(record)

    failed_case_ids = [case["case_id"] for case in cases if not case["correctness"]["passed"]]
    run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run["status"] = "failed" if failed_case_ids else "passed"
    run["case_count"] = len(cases)
    run["failed_case_ids"] = failed_case_ids

    if args.output is not None:
        output_format = _resolve_output_format(args.output, args.output_format)
        _write_output(
            args.output,
            output_format=output_format,
            overwrite=args.overwrite,
            run=run,
            cases=cases,
        )
        print(f"structured output: {args.output} ({output_format})")
    print(
        f"overall correctness: {'PASS' if not failed_case_ids else 'FAIL'} "
        f"({len(cases) - len(failed_case_ids)}/{len(cases)} cases)"
    )
    if failed_case_ids:
        raise RuntimeError(
            "paged decode benchmark correctness failed: " + ", ".join(failed_case_ids)
        )


if __name__ == "__main__":
    main()
