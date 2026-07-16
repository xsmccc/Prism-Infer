"""P6 外部框架 offline benchmark 校验与 Prism baseline 对比。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prism_infer.analysis.benchmark_schema import validate_benchmark_record
from prism_infer.analysis.pareto_summary import stable_prefix_lengths


SUPPORTED_EXTERNAL_SCHEMA_VERSIONS = (1, 2)
COMPARISON_PROFILES = ("diagnostic_matched", "best_stable")


def validate_external_record(record: Mapping[str, Any]) -> None:
    """校验 P6/P7 external benchmark 的关键证据字段。"""

    schema_version = record.get("schema_version")
    if schema_version not in SUPPORTED_EXTERNAL_SCHEMA_VERSIONS:
        raise ValueError(
            "external record schema_version must be one of "
            f"{SUPPORTED_EXTERNAL_SCHEMA_VERSIONS}"
        )
    if record.get("record_type") != "external_system_benchmark":
        raise ValueError("external record_type must be external_system_benchmark")
    for section in (
        "environment",
        "model",
        "backend",
        "workload",
        "measurement",
        "correctness",
        "timing_ms",
        "throughput",
        "memory_mb",
    ):
        if not isinstance(record.get(section), Mapping):
            raise ValueError(f"external record missing object section: {section}")
    environment = record["environment"]
    for key in ("framework", "framework_version", "framework_source_commit", "gpu"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"external environment.{key} must be a non-empty string")
    measurement = record["measurement"]
    if measurement.get("repeat", 0) < 1 or measurement.get("warmup", -1) < 0:
        raise ValueError("external warmup/repeat are invalid")
    if measurement.get("cuda_synchronize_timing") is not True:
        raise ValueError("external benchmark must use CUDA synchronized timing")
    correctness = record["correctness"]
    if correctness.get("outputs_identical_across_repeats") is not True:
        raise ValueError("external greedy outputs are not stable across repeats")
    token_ids = correctness.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("external correctness.token_ids must be non-empty")
    for path in (
        ("timing_ms", "end_to_end"),
        ("timing_ms", "engine_ttft"),
        ("timing_ms", "decode_tpot"),
        ("throughput", "e2e_output_tokens_per_s"),
    ):
        summary = record[path[0]].get(path[1])
        if not isinstance(summary, Mapping) or summary.get("count", 0) < 1:
            raise ValueError(f"external {path[0]}.{path[1]} is invalid")
    memory = record["memory_mb"]
    memory_summary = memory.get("peak_allocated", memory.get("process_used"))
    if not isinstance(memory_summary, Mapping) or memory_summary.get("count", 0) < 1:
        raise ValueError("external memory evidence is invalid")
    if schema_version >= 2:
        protocol = record.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError("schema-v2 external record requires protocol")
        if protocol.get("name") != "p7.1_external_offline_v2":
            raise ValueError("unsupported schema-v2 external protocol")
        if protocol.get("comparison_profile") not in COMPARISON_PROFILES:
            raise ValueError("invalid external comparison_profile")
        for key in ("harness_git_commit", "process_scope"):
            if not isinstance(protocol.get(key), str) or not protocol[key]:
                raise ValueError(f"external protocol.{key} must be non-empty")
        for key in ("harness_git_dirty", "framework_source_dirty"):
            if not isinstance(protocol.get(key), bool):
                raise ValueError(f"external protocol.{key} must be a bool")
        command = protocol.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ValueError("external protocol.command must be a non-empty argv list")
        for key in ("gpu_uuid", "driver", "compute_capability"):
            if not isinstance(environment.get(key), str) or not environment[key]:
                raise ValueError(f"external environment.{key} must be non-empty")
        backend = record["backend"]
        for key in ("execution", "cudagraph_mode", "compilation_mode"):
            if not isinstance(backend.get(key), str) or not backend[key]:
                raise ValueError(f"external backend.{key} must be non-empty")
        sampling = record.get("sampling")
        if not isinstance(sampling, Mapping):
            raise ValueError("schema-v2 external record requires sampling")
        if sampling.get("temperature") != 0.0:
            raise ValueError("external benchmark must use temperature=0")
        if sampling.get("ignore_eos") is not True:
            raise ValueError("external benchmark must use ignore_eos=true")
        if sampling.get("max_tokens") != record["workload"].get("max_tokens"):
            raise ValueError("external sampling/workload max_tokens mismatch")


def load_external_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """读取并校验外部框架 JSON records。"""

    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_external_record(record)
        records.append(record)
    return records


def _key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    workload = record["workload"]
    return (
        workload["manifest_sha256"],
        workload["case_id"],
        workload["num_requests"],
        workload["max_tokens"],
    )


def _execution_profile_matches(
    prism: Mapping[str, Any],
    external: Mapping[str, Any],
    profile: str,
) -> bool:
    prism_execution = prism["mode"]["execution"]
    external_execution = external["backend"]["execution"]
    if profile == "diagnostic_matched":
        return (
            prism_execution == "eager"
            and external_execution == "eager"
            and prism["model"].get("chunked_prefill_enabled") is False
            and external["backend"].get("chunked_prefill") is False
            and external["backend"].get("async_scheduling") is False
        )
    if profile == "best_stable":
        return (
            prism_execution == "cuda_graph"
            and external_execution == "cuda_graph"
            and external["backend"].get("cudagraph_mode") not in (None, "NONE")
        )
    return False


def _p7_comparability_checks(
    prism: Mapping[str, Any],
    external: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, bool]:
    """Return explicit gates instead of silently comparing near-matching runs."""

    prism_workload = prism["workload"]
    external_workload = external["workload"]
    prism_model = prism["model"]
    external_model = external["model"]
    prism_measurement = prism["measurement"]
    external_measurement = external["measurement"]
    prism_environment = prism["environment"]
    external_environment = external["environment"]
    protocol = external["protocol"]
    prism_sampling = prism.get("sampling", {})
    external_sampling = external.get("sampling", {})
    return {
        "comparison_profile": protocol.get("comparison_profile") == profile,
        "prompt_tokens": (
            prism_workload.get("prompt_tokens")
            == external_workload.get("prompt_tokens")
        ),
        "model_config": (
            prism_model.get("config_sha256")
            == external_model.get("config_sha256")
            and prism_model.get("config_sha256") not in (None, "unknown")
        ),
        "dtype": prism_model.get("dtype") == external_model.get("dtype"),
        "tensor_parallel_size": (
            prism_model.get("tensor_parallel_size")
            == external_model.get("tensor_parallel_size")
        ),
        "max_model_len": (
            prism_model.get("max_model_len")
            == external_model.get("max_model_len")
        ),
        "max_num_batched_tokens": (
            prism_model.get("max_num_batched_tokens")
            == external_model.get("max_num_batched_tokens")
        ),
        "max_num_seqs": (
            prism_model.get("max_num_seqs")
            == external_model.get("max_num_seqs")
        ),
        "block_size": (
            prism_model.get("kvcache_block_size")
            == external["backend"].get("block_size")
        ),
        "kv_pool_bytes": (
            prism["kv_cache"].get("bytes")
            == external_model.get("kv_cache_memory_bytes")
        ),
        "prefix_caching": (
            prism_model.get("prefix_caching_enabled") is False
            and external["backend"].get("prefix_caching") is False
        ),
        "traffic": (
            prism.get("traffic", {}).get("kind") == "offline_closed_loop"
            and external_workload.get("traffic") == "offline_closed_loop"
        ),
        "preprocessing_scope": (
            prism_workload.get("preprocessing_included_in_e2e") is True
            and external_workload.get("preprocessing_included_in_e2e") is True
        ),
        "output_decoding_scope": (
            prism_workload.get("output_decoding_included_in_e2e") is False
            and external_workload.get("output_decoding_included_in_e2e") is False
        ),
        "warmup": prism_measurement.get("warmup") == external_measurement.get("warmup"),
        "repeat": prism_measurement.get("repeat") == external_measurement.get("repeat"),
        "cuda_synchronized": (
            prism_measurement.get("cuda_synchronize_timing") is True
            and external_measurement.get("cuda_synchronize_timing") is True
        ),
        "sampling": prism_sampling == external_sampling,
        "execution_profile": _execution_profile_matches(
            prism, external, profile
        ),
        "same_gpu": (
            prism_environment.get("gpu_uuid")
            == external_environment.get("gpu_uuid")
            and prism_environment.get("gpu_uuid") not in (None, "unknown")
        ),
        "clean_prism": prism_environment.get("git_dirty") is False,
        "clean_harness": (
            protocol.get("harness_git_dirty") is False
            and protocol.get("harness_git_commit")
            == prism_environment.get("git_commit")
        ),
        "clean_external_source": protocol.get("framework_source_dirty") is False,
    }


def compare_external_records(
    prism_records: Sequence[Mapping[str, Any]],
    external_records: Sequence[Mapping[str, Any]],
    *,
    prism_mode: str = "off_eager",
    prism_keep_ratio: float = 0.5,
    comparison_profile: str | None = None,
) -> list[dict[str, Any]]:
    """将 external records 与唯一匹配的 Prism benchmark cell 比较。"""

    baselines: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for record in prism_records:
        validate_benchmark_record(record)
        if record["mode"]["name"] != prism_mode:
            continue
        if record["mode"]["visual_pruning_keep_ratio"] != prism_keep_ratio:
            continue
        key = _key(record)
        if key in baselines:
            raise ValueError(f"duplicate Prism external-comparison baseline: {key}")
        baselines[key] = record

    rows: list[dict[str, Any]] = []
    for external in external_records:
        validate_external_record(external)
        external_profile = (
            external.get("protocol", {}).get("comparison_profile", "legacy_p6")
        )
        if comparison_profile is not None and external_profile != comparison_profile:
            continue
        key = _key(external)
        if key not in baselines:
            raise ValueError(f"missing matching Prism baseline for external key={key}")
        prism = baselines[key]
        prism_tokens = prism["correctness"]["token_ids"]
        external_tokens = external["correctness"]["token_ids"]
        prefixes = stable_prefix_lengths(prism_tokens, external_tokens)
        prism_tpot = prism["timing_ms"]["decode_step"]["median"]
        external_tpot = external["timing_ms"]["decode_tpot"]["median"]
        prism_throughput = prism["throughput"]["e2e_output_tokens_per_s"]["median"]
        external_throughput = external["throughput"]["e2e_output_tokens_per_s"]["median"]
        prism_peak = prism["memory_mb"]["peak_allocated"]["median"]
        external_memory = external["memory_mb"]
        external_peak = external_memory.get(
            "peak_allocated", external_memory.get("process_used")
        )["median"]
        prism_memory_measurement = prism["memory_mb"].get(
            "measurement", "torch_cuda_allocator"
        )
        external_memory_measurement = external["memory_mb"].get(
            "measurement", "torch_cuda_allocator"
        )
        memory_comparable = prism_memory_measurement == external_memory_measurement
        if min(prism_tpot, prism_throughput, prism_peak) <= 0:
            raise ValueError(f"Prism baseline has a non-positive denominator: {key}")
        input_token_exact = (
            prism["workload"]["prompt_tokens"]
            == external["workload"]["prompt_tokens"]
        )
        if external["schema_version"] >= 2:
            checks = _p7_comparability_checks(
                prism,
                external,
                profile=external_profile,
            )
        else:
            checks = {"prompt_tokens": input_token_exact}
        non_comparable_reasons = [
            name for name, passed in checks.items() if not passed
        ]
        performance_comparable = not non_comparable_reasons
        rows.append(
            {
                "comparison_profile": external_profile,
                "case_id": external["workload"]["case_id"],
                "num_requests": external["workload"]["num_requests"],
                "max_tokens": external["workload"]["max_tokens"],
                "prompt_tokens_prism": prism["workload"]["prompt_tokens"],
                "prompt_tokens_external": external["workload"]["prompt_tokens"],
                "input_token_exact": input_token_exact,
                "comparability_checks": checks,
                "non_comparable_reasons": non_comparable_reasons,
                "performance_comparable": performance_comparable,
                "prism_mode": prism["mode"]["name"],
                "prism_execution": prism["mode"].get("execution", "unknown"),
                "external_execution": external["backend"].get("execution"),
                "external_cudagraph_mode": external["backend"].get(
                    "cudagraph_mode"
                ),
                "framework": external["environment"]["framework"],
                "framework_version": external["environment"]["framework_version"],
                "framework_source_commit": external["environment"][
                    "framework_source_commit"
                ],
                "stable_prefix_per_request": prefixes,
                "token_exact": prism_tokens == external_tokens,
                "prism_tpot_median_ms": prism_tpot,
                "external_tpot_median_ms": external_tpot,
                "external_to_prism_tpot_ratio": (
                    external_tpot / prism_tpot
                    if performance_comparable
                    else None
                ),
                "prism_e2e_output_tokens_per_s": prism_throughput,
                "external_e2e_output_tokens_per_s": external_throughput,
                "external_to_prism_throughput_ratio": (
                    external_throughput / prism_throughput
                    if performance_comparable
                    else None
                ),
                "prism_peak_allocated_mib": prism_peak,
                "external_peak_allocated_mib": external_peak,
                "memory_comparable": memory_comparable,
                "prism_memory_measurement": prism_memory_measurement,
                "external_memory_measurement": external_memory_measurement,
                "external_to_prism_peak_memory_ratio": (
                    external_peak / prism_peak
                    if performance_comparable and memory_comparable
                    else None
                ),
                "prism_physical_prompt_tokens": prism.get("kv_cache", {}).get(
                    "physical_prompt_tokens"
                ),
                "prism_active_prompt_bytes": prism.get("kv_cache", {}).get(
                    "active_prompt_bytes"
                ),
            }
        )
    if comparison_profile is not None and not rows:
        raise ValueError(
            f"no external records matched comparison_profile={comparison_profile!r}"
        )
    return rows


def render_external_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """渲染 external vs Prism 紧凑表格。"""

    lines = [
        "| Profile | Prism mode | Framework | Case | Prompt P/E | Comparable | Why not | Stable prefix | Exact | TPOT P/E | E/P TPOT | "
        "Throughput P/E | E/P throughput | Memory MiB P/E | Memory comparable |",
        "|---|---|---|---|---:|:---:|---|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        tpot_ratio = (
            f"{row['external_to_prism_tpot_ratio']:.3f}x"
            if row["performance_comparable"]
            else "n/a"
        )
        throughput_ratio = (
            f"{row['external_to_prism_throughput_ratio']:.3f}x"
            if row["performance_comparable"]
            else "n/a"
        )
        reasons = ",".join(row.get("non_comparable_reasons", [])) or "-"
        lines.append(
            f"| {row.get('comparison_profile', 'legacy_p6')} | "
            f"{row.get('prism_mode', 'unknown')} | "
            f"{row['framework']} {row['framework_version']} | {row['case_id']} | "
            f"{row['prompt_tokens_prism']}/"
            f"{row['prompt_tokens_external']} | "
            f"{'yes' if row['performance_comparable'] else 'no'} | "
            f"{reasons} | "
            f"{row['stable_prefix_per_request']} | "
            f"{'yes' if row['token_exact'] else 'no'} | "
            f"{row['prism_tpot_median_ms']:.3f}/{row['external_tpot_median_ms']:.3f} | "
            f"{tpot_ratio} | "
            f"{row['prism_e2e_output_tokens_per_s']:.3f}/"
            f"{row['external_e2e_output_tokens_per_s']:.3f} | "
            f"{throughput_ratio} | "
            f"{row['prism_peak_allocated_mib']:.1f}/"
            f"{row['external_peak_allocated_mib']:.1f} | "
            f"{'yes' if row['memory_comparable'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"
