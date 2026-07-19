"""P6.3-B ``torch.compile`` region preflight 与固定 shape benchmark。

脚本复用 P6 deterministic single-image workload，分别调查 decoder layer、
language-model decode 和完整 VisionEncoder。Dynamo 诊断使用 eager FX backend；
只有未被 ``--skip-inductor`` 排除的 region 才执行 Inductor cold/steady benchmark。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch
from torch._inductor import config as inductor_config


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_system import (
    DEFAULT_MANIFEST,
    _add_requests,
)
from benchmarks.harness import (
    collect_git_metadata,
    expand_case_batch,
    find_workload_case,
    materialize_requests,
)
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.benchmark_schema import load_workload_manifest
from prism_infer.analysis.compile_preflight import (
    COMPILE_PREFLIGHT_REGIONS,
    COMPILE_PREFLIGHT_SCHEMA_VERSION,
    build_latency_stats,
    compare_tensor_outputs,
    run_recompile_probe,
    summarize_explain_output,
    validate_compile_preflight_record,
)
from prism_infer.engine.contracts import DeviceModelInputs
from prism_infer.utils.context import reset_context


@dataclass
class RegionWorkload:
    """一个 compile region 的 callable 和 shape matrix。"""

    function: Callable[..., Any]
    explain_args: tuple[Any, ...]
    invocations: list[tuple[Any, ...]]
    benchmark_args: tuple[Any, ...]
    boundary: str


def _sync_measure(
    function: Callable[..., Any],
    args: tuple[Any, ...],
) -> tuple[Any, float]:
    """在显式 CUDA 同步边界内测量一次调用。"""

    torch.cuda.synchronize()
    start = perf_counter()
    output = function(*args)
    torch.cuda.synchronize()
    return output, (perf_counter() - start) * 1000.0


def _benchmark_inductor_configured(
    workload: RegionWorkload,
    *,
    mode: str,
    warmup: int,
    repeat: int,
    fullgraph: bool,
    emulate_precision_casts: bool,
    force_same_precision: bool,
) -> dict[str, Any]:
    """记录 Inductor 首次调用、steady latency、显存和 eager correctness。"""

    torch._dynamo.reset()
    eager_reference, _ = _sync_measure(workload.function, workload.benchmark_args)
    torch.cuda.reset_peak_memory_stats()
    try:
        compiled = torch.compile(
            workload.function,
            backend="inductor",
            mode=mode,
            fullgraph=fullgraph,
            dynamic=False,
        )
        compiled_output, first_call_ms = _sync_measure(
            compiled,
            workload.benchmark_args,
        )
        correctness = compare_tensor_outputs(eager_reference, compiled_output)

        for _ in range(warmup):
            _sync_measure(compiled, workload.benchmark_args)
        compiled_latencies = [
            _sync_measure(compiled, workload.benchmark_args)[1] for _ in range(repeat)
        ]
        eager_latencies = [
            _sync_measure(workload.function, workload.benchmark_args)[1] for _ in range(repeat)
        ]
        compiled_stats = build_latency_stats(compiled_latencies)
        record = {
            "attempted": True,
            "status": "pass",
            "backend": "inductor",
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": False,
            "emulate_precision_casts": emulate_precision_casts,
            "force_same_precision": force_same_precision,
            "warmup": warmup,
            "repeat": repeat,
            "first_call_ms": first_call_ms,
            "compile_overhead_ms": max(
                first_call_ms - float(compiled_stats["median"]),
                0.0,
            ),
            "allocated_memory_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_memory_mb": torch.cuda.memory_reserved() / (1024**2),
            "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "eager_ms": build_latency_stats(eager_latencies),
            "compiled_ms": compiled_stats,
            "correctness": correctness,
        }
        if correctness["max_abs_diff"] >= 1e-5:
            record["status"] = "failed"
            record["error"] = (
                "same-precision correctness threshold failed: "
                f"max_abs_diff={correctness['max_abs_diff']:.6e} >= 1e-5"
            )
        return record
    except Exception as error:
        return {
            "attempted": True,
            "status": "failed",
            "backend": "inductor",
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": False,
            "emulate_precision_casts": emulate_precision_casts,
            "force_same_precision": force_same_precision,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        torch._dynamo.reset()


def _benchmark_inductor(
    workload: RegionWorkload,
    *,
    mode: str,
    warmup: int,
    repeat: int,
    fullgraph: bool,
    emulate_precision_casts: bool,
    force_same_precision: bool,
) -> dict[str, Any]:
    """在显式精度配置下执行一次隔离的 Inductor benchmark。"""

    with inductor_config.patch(
        {
            "emulate_precision_casts": emulate_precision_casts,
            "force_same_precision": force_same_precision,
        }
    ):
        return _benchmark_inductor_configured(
            workload,
            mode=mode,
            warmup=warmup,
            repeat=repeat,
            fullgraph=fullgraph,
            emulate_precision_casts=emulate_precision_casts,
            force_same_precision=force_same_precision,
        )


def _prepare_engine(
    args: argparse.Namespace,
) -> tuple[LLM, list[Any], DeviceModelInputs]:
    """运行一次真实 single-image prefill，并保留下一步 decode 输入/context。"""

    manifest = load_workload_manifest(args.manifest)
    source_case = find_workload_case(manifest, "single_image_448")
    case, _, _ = expand_case_batch(source_case, args.max_batch_size)
    llm = LLM(
        args.model,
        enforce_eager=True,
        compression_mode="off",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
        enable_chunked_prefill=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    requests = materialize_requests(case, repo_root=REPO_ROOT)
    _add_requests(llm, requests, sampling)

    prefill_seqs, is_prefill, _, _, _ = llm.scheduler.schedule()
    if not is_prefill or len(prefill_seqs) != args.max_batch_size:
        raise RuntimeError("compile preflight failed to schedule the expected prefill batch")
    token_ids = llm.model_runner.run(prefill_seqs, True)
    llm.scheduler.postprocess(prefill_seqs, token_ids)

    decode_seqs, is_prefill, _, _, _ = llm.scheduler.schedule()
    if is_prefill or len(decode_seqs) != args.max_batch_size:
        raise RuntimeError("compile preflight failed to schedule the expected decode batch")
    decode_inputs = llm.model_runner.prepare_decode(decode_seqs)
    return llm, decode_seqs, decode_inputs


def _slice_decode_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """按 engine flatten batch 语义切片 decode tensor。"""

    if tensor.ndim == 2 and tensor.shape[0] == 3:
        return tensor[:, :batch_size]
    return tensor[:batch_size]


def _build_decoder_workload(
    llm: LLM,
    decode_inputs: DeviceModelInputs,
    batch_sizes: list[int],
    boundary: str,
    benchmark_batch_size: int,
) -> RegionWorkload:
    """构造第 0 个真实 decoder layer 的 engine-decode 调用。"""

    language_model = llm.model_runner.model.model.language_model
    layer = language_model.layers[0]
    input_ids = decode_inputs.input_ids
    position_ids = llm.model_runner._as_mrope_decode_positions(decode_inputs.position_ids)
    hidden_states = language_model.embed_tokens(input_ids)
    rope_position_ids = position_ids[:, None, :]
    cos, sin = language_model.rotary_emb(hidden_states, rope_position_ids)

    def full_layer(
        hidden: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        return layer(hidden, (rotary_cos, rotary_sin))

    def attention(
        hidden: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        return layer.self_attn(hidden, (rotary_cos, rotary_sin))

    def qkv_prep(
        hidden: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return layer.self_attn._forward_engine_qkv(
            hidden,
            (rotary_cos, rotary_sin),
        )

    full_invocations = [
        (
            hidden_states[:batch_size],
            cos[:, :batch_size],
            sin[:, :batch_size],
        )
        for batch_size in batch_sizes
    ]
    if boundary == "full_layer":
        function = full_layer
        invocations = full_invocations
    elif boundary == "attention":
        function = attention
        invocations = full_invocations
    elif boundary == "qkv_prep":
        function = qkv_prep
        invocations = [
            (hidden, rotary_cos.squeeze(0), rotary_sin.squeeze(0))
            for hidden, rotary_cos, rotary_sin in full_invocations
        ]
    elif boundary == "rmsnorm":
        function = layer.input_layernorm
        invocations = [(values[0],) for values in full_invocations]
    elif boundary == "mlp":
        function = layer.mlp
        invocations = [(values[0],) for values in full_invocations]
    else:
        raise ValueError(f"unsupported decoder boundary: {boundary}")
    invocations.append(invocations[0])
    benchmark_args = invocations[batch_sizes.index(benchmark_batch_size)]
    return RegionWorkload(
        function=function,
        explain_args=invocations[-2],
        invocations=invocations,
        benchmark_args=benchmark_args,
        boundary=boundary,
    )


def _build_language_decode_workload(
    llm: LLM,
    decode_inputs: DeviceModelInputs,
    batch_sizes: list[int],
    benchmark_batch_size: int,
) -> RegionWorkload:
    """构造完整 language-model decode 调用，不包含 logits/sampler。"""

    model = llm.model_runner.model

    def language_model_decode(
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return model(input_ids=input_ids, position_ids=position_ids)

    invocations = [
        (
            decode_inputs.input_ids[:batch_size],
            _slice_decode_tensor(decode_inputs.position_ids, batch_size),
        )
        for batch_size in batch_sizes
    ]
    invocations.append(invocations[0])
    return RegionWorkload(
        function=language_model_decode,
        explain_args=invocations[-2],
        invocations=invocations,
        benchmark_args=invocations[batch_sizes.index(benchmark_batch_size)],
        boundary="full_language_model_decode",
    )


def _build_vision_workload(
    llm: LLM,
    decode_seqs: list[Any],
    boundary: str,
) -> RegionWorkload:
    """构造 196/784 patch 的 VisionEncoder shape matrix。"""

    sequence = decode_seqs[0]
    pixel_values = sequence.pixel_values.cuda(non_blocking=True)
    grid_thw = sequence.image_grid_thw.cuda(non_blocking=True)
    if pixel_values.shape[0] < 784:
        raise RuntimeError(
            "single_image_448 preflight expects at least 784 processor patches, "
            f"got {pixel_values.shape[0]}"
        )
    if grid_thw.shape != (1, 3):
        raise RuntimeError(f"unexpected image grid shape: {list(grid_thw.shape)}")
    small_grid = grid_thw.new_tensor([[1, 14, 14]])
    small_pixels = pixel_values[:196].contiguous()
    full_pixels = pixel_values[:784].contiguous()
    full_grid = grid_thw.new_tensor([[1, 28, 28]])
    visual = llm.model_runner.model.model.visual

    def full_vision_encoder(
        pixels: torch.Tensor,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return visual(pixels, grid)

    small_args = (small_pixels, small_grid)
    full_args = (full_pixels, full_grid)
    if boundary == "full_encoder":
        function = full_vision_encoder
    elif boundary == "tensor_region":
        small_args = visual.prepare_tensor_region_inputs(*small_args)
        full_args = visual.prepare_tensor_region_inputs(*full_args)
        function = visual.forward_tensor_region
    else:
        raise ValueError(f"unsupported vision boundary: {boundary}")
    return RegionWorkload(
        function=function,
        explain_args=full_args,
        invocations=[small_args, full_args, small_args],
        benchmark_args=full_args,
        boundary=boundary,
    )


def _build_workload(
    region: str,
    llm: LLM,
    decode_seqs: list[Any],
    decode_inputs: DeviceModelInputs,
    batch_sizes: list[int],
    decoder_boundary: str,
    vision_boundary: str,
    benchmark_batch_size: int,
) -> RegionWorkload:
    """按 region 名称构造 preflight workload。"""

    if region == "decoder_layer":
        return _build_decoder_workload(
            llm,
            decode_inputs,
            batch_sizes,
            decoder_boundary,
            benchmark_batch_size,
        )
    if region == "language_model_decode":
        return _build_language_decode_workload(
            llm,
            decode_inputs,
            batch_sizes,
            benchmark_batch_size,
        )
    if region == "vision_encoder":
        return _build_vision_workload(llm, decode_seqs, vision_boundary)
    raise ValueError(f"unsupported compile preflight region: {region}")


def _run_preflight(
    args: argparse.Namespace,
    workload: RegionWorkload,
) -> dict[str, Any]:
    """执行 explain、recompile probe 和可选 Inductor benchmark。"""

    with torch.inference_mode():
        explain_output = torch._dynamo.explain(workload.function)(*workload.explain_args)
        dynamo = summarize_explain_output(explain_output)
        _, recompile = run_recompile_probe(
            workload.function,
            workload.invocations,
            dynamic=False,
        )

        if args.skip_inductor:
            benchmark = {
                "attempted": False,
                "status": "skipped",
                "reason": "Inductor disabled by --skip-inductor for diagnostic preflight",
            }
        else:
            benchmark = _benchmark_inductor(
                workload,
                mode=args.compile_mode,
                warmup=args.warmup,
                repeat=args.repeat,
                fullgraph=dynamo["graph_break_count"] == 0,
                emulate_precision_casts=args.emulate_precision_casts,
                force_same_precision=args.force_same_precision,
            )
    return {
        "dynamo": dynamo,
        "recompile": recompile,
        "benchmark": benchmark,
    }


def _parse_batch_sizes(value: str, max_batch_size: int) -> list[int]:
    """解析递增且不超过最大 batch 的 shape matrix。"""

    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 or item > max_batch_size for item in values):
        raise ValueError("batch sizes must be within [1, max_batch_size]")
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--region", choices=COMPILE_PREFLIGHT_REGIONS, required=True)
    parser.add_argument(
        "--decoder-boundary",
        choices=("full_layer", "attention", "qkv_prep", "rmsnorm", "mlp"),
        default="full_layer",
    )
    parser.add_argument(
        "--vision-boundary",
        choices=("full_encoder", "tensor_region"),
        default="full_encoder",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument(
        "--benchmark-batch-size",
        type=int,
        help="fixed batch used by Inductor timing; defaults to max batch",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead"), default="default")
    parser.add_argument("--emulate-precision-casts", action="store_true")
    parser.add_argument("--force-same-precision", action="store_true")
    parser.add_argument("--skip-inductor", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=16)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for compile preflight")
    if args.max_batch_size < 1:
        raise SystemExit("--max-batch-size must be positive")
    if args.warmup < 1 or args.repeat < 1:
        raise SystemExit("--warmup and --repeat must be positive")
    batch_sizes = _parse_batch_sizes(args.batch_sizes, args.max_batch_size)
    if batch_sizes[-1] != args.max_batch_size:
        raise SystemExit("--batch-sizes must include --max-batch-size")
    benchmark_batch_size = args.benchmark_batch_size or args.max_batch_size
    if benchmark_batch_size not in batch_sizes:
        raise SystemExit("--benchmark-batch-size must be present in --batch-sizes")

    git = collect_git_metadata(REPO_ROOT, strict=True)
    llm = None
    try:
        llm, decode_seqs, decode_inputs = _prepare_engine(args)
        workload = _build_workload(
            args.region,
            llm,
            decode_seqs,
            decode_inputs,
            batch_sizes,
            args.decoder_boundary,
            args.vision_boundary,
            benchmark_batch_size,
        )
        evidence = _run_preflight(args, workload)
        record = {
            "schema_version": COMPILE_PREFLIGHT_SCHEMA_VERSION,
            "record_type": "compile_preflight",
            "region": args.region,
            "environment": {
                "torch_version": torch.__version__,
                "cuda_version": str(torch.version.cuda),
                "gpu": torch.cuda.get_device_name(0),
                "git_commit": git.commit,
                "git_dirty": git.dirty,
            },
            "config": {
                "model": args.model,
                "compression": "off",
                "attention": "paged_triton",
                "batch_sizes": batch_sizes,
                "benchmark_batch_size": benchmark_batch_size,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "num_kvcache_blocks": args.num_kvcache_blocks,
                "kvcache_block_size": args.kvcache_block_size,
                "region_boundary": workload.boundary,
            },
            "inputs": evidence["recompile"]["invocation_shapes"],
            **evidence,
        }
        validate_compile_preflight_record(record)
        rendered = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
            print(f"wrote compile preflight record to {output_path}")
        else:
            print(rendered)
    finally:
        reset_context()
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
