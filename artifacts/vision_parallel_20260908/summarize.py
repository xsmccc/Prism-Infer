"""Summarize explicitly selected Vision-DP evidence without changing raw records.

The initial TP2 pair is fixed at job 9764. Supply both --final-replicated and
--final-data to add a later final comparison; job numbers are never ranked.
Relative optional input paths are relative to --raw-dir. Only the standard
library is required. MuirBench scoring here is limited to this Qwen3-VL sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.vision_parallel_report import summarize_requests  # noqa: E402

EOS_TOKEN_ID = 151645
CHOICE_TOKENS = {32: "A", 33: "B", 34: "C", 35: "D", 36: "E"}
PHASE_METRICS = (
    "ttft_median_ms",
    "tpot_median_ms",
    "e2e_median_ms",
    "output_tokens_per_s",
    "requests_per_s",
    "duration_s",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_name(path: Path) -> str:
    return "/".join(path.parts[-2:])


def selected_path(raw_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else raw_dir / path


def percent_changes(reference: dict, candidate: dict) -> dict:
    return {
        key: 100 * (candidate[key] / reference[key] - 1)
        for key in PHASE_METRICS
        if reference.get(key) not in (None, 0)
        and candidate.get(key) is not None
        and (
            key != "duration_s"
            or reference.get("duration_scope")
            == candidate.get("duration_scope")
            == "single_concurrent_burst"
        )
    }


def summarize_features(path: Path) -> dict:
    raw = read_json(path)
    by_case = {}
    for rank in raw["ranks"]:
        for case in rank["cases"]:
            by_case.setdefault(case["case"], []).append((rank["rank"], case))
    cases = {}
    for name, rank_cases in by_case.items():
        timing = {}
        for mode in ("replicated", "data"):
            times = [case[f"{mode}_ms"] for _, case in rank_cases]
            slow_repeats = [max(values) for values in zip(*times, strict=True)]
            timing[mode] = {
                "slow_rank_ms_per_repeat": slow_repeats,
                "slow_rank_median_ms": median(slow_repeats),
                "max_of_rank_medians_ms": max(median(values) for values in times),
            }
        baseline = timing["replicated"]["slow_rank_median_ms"]
        candidate = timing["data"]["slow_rank_median_ms"]
        features = {}
        for index in range(len(rank_cases[0][1]["features"])):
            feature_name = "main" if index == 0 else f"deepstack{index - 1}"
            errors = [case["features"][index] for _, case in rank_cases]
            features[feature_name] = {
                "max_relative_l2_across_ranks": max(item["relative_l2"] for item in errors),
                "max_abs_across_ranks": max(item["max_abs"] for item in errors),
                "min_equal_fraction_across_ranks": min(item["equal_fraction"] for item in errors),
            }
        cases[name] = {
            "grid": rank_cases[0][1]["grid"],
            "timing": timing,
            "speedup_replicated_over_data": baseline / candidate,
            "latency_reduction_percent": 100 * (1 - candidate / baseline),
            "feature_drift": features,
        }
    return {
        "source": source_name(path),
        "model": raw["model"],
        "timing_definition": "median over repeats of the maximum elapsed time across ranks",
        "scope": "Vision-only; not end-to-end speedup or task-quality equivalence",
        "cases": cases,
    }


def summarize_diagnosis(path: Path) -> dict:
    raw = read_json(path)
    kept = {
        "patch_embed",
        "position_embedding",
        "position_plus_patch",
        "rope_cos",
        "rope_sin",
        "block00.norm1",
        "block00.qkv",
        "block00.attention_projection",
        "block00",
        "block08",
        "block16",
        "block24",
        "block25",
        "block26",
        "main_merger",
    }
    return {
        "source": source_name(path),
        "communication": raw["communication"],
        "fp32_weight_source": raw["fp32_weight_source"],
        "boundary": (
            "The attention_projection hook observes its output, not its input. "
            "It does not identify a specific SDPA or cuBLAS algorithm as the cause."
        ),
        "runs": [
            {
                "case": run["case"],
                "precision": run["precision"],
                "assignments": run["assignments"],
                "first_observed_unequal_stage": run["first_unequal_stage"],
                "stage_errors": {
                    name: {key: error[key] for key in ("relative_l2", "max_abs", "equal_fraction")}
                    for name, error in run["stage_errors"].items()
                    if name in kept or name.startswith("deepstack")
                },
            }
            for run in raw["runs"]
        ],
    }


def compact_report(raw: dict, path: Path) -> dict:
    result = {"source": source_name(path)}
    for key in (
        "status",
        "error",
        "git",
        "model_path",
        "model_revision",
        "model_load_ms",
        "engine_options",
        "input",
        "sampling",
        "protocol",
        "kv_cache",
        "configured_kv_capacity",
        "partial_prefix_validation",
    ):
        if key in raw:
            result[key] = raw[key]
    result["gpus"] = [
        {key: gpu.get(key) for key in ("name", "gpu_uuid", "driver", "compute_capability")}
        for gpu in raw.get("gpus", [])
    ]
    by_phase: dict[str, list[dict]] = {}
    for row in raw.get("requests", []):
        by_phase.setdefault(row["phase"], []).append(row)
    result["phase_summaries"] = {
        phase: summarize_requests(rows, concurrent=phase.startswith("concurrent_"))
        for phase, rows in by_phase.items()
    }
    result["phase_summaries_source"] = "recomputed from raw requests, not raw phase_summaries"
    return result


def index_requests(rows: list[dict]) -> dict:
    # Sequential repeats call _run_requests separately and therefore reuse
    # phase_request_index=0. Pair by occurrence in each recorded phase instead.
    occurrences: Counter = Counter()
    indexed = {}
    for row in rows:
        phase = row["phase"]
        indexed[(phase, occurrences[phase])] = row
        occurrences[phase] += 1
    return indexed


def summarize_tp2_pair(reference_path: Path, candidate_path: Path) -> dict:
    reference = read_json(reference_path)
    candidate = read_json(candidate_path)
    result = {
        "replicated": compact_report(reference, reference_path),
        "data": compact_report(candidate, candidate_path),
        "change_definition": "100 * (data / replicated - 1); negative latency is better",
        "scope": "same TP2 implementation modes; finite workloads, not saturation throughput",
        "phase_comparison": {},
    }
    if reference.get("status") != "complete" or candidate.get("status") != "complete":
        result["comparison_status"] = "incomplete_not_a_performance_result"
        return result
    left_rows = index_requests(reference["requests"])
    right_rows = index_requests(candidate["requests"])
    result["pairing"] = "phase and within-phase occurrence in the recorded fixed request stream"
    result["request_keys_equal"] = left_rows.keys() == right_rows.keys()
    result["same_input_fixture"] = reference["input"] == candidate["input"]
    result["same_sampling"] = reference["sampling"] == candidate["sampling"]
    result["same_kv_bytes"] = reference["kv_cache"] == candidate["kv_cache"]
    option_keys = reference["engine_options"].keys() | candidate["engine_options"].keys()
    result["engine_option_differences"] = {
        key: [reference["engine_options"].get(key), candidate["engine_options"].get(key)]
        for key in sorted(option_keys)
        if reference["engine_options"].get(key) != candidate["engine_options"].get(key)
    }
    for phase, before in result["replicated"]["phase_summaries"].items():
        if phase not in result["data"]["phase_summaries"]:
            continue
        after = result["data"]["phase_summaries"][phase]
        pairs = [
            (row, right_rows[key])
            for key, row in left_rows.items()
            if key[0] == phase and key in right_rows
        ]
        result["phase_comparison"][phase] = {
            "metric_change_percent": percent_changes(before, after),
            "paired_requests": len(pairs),
            "prompt_token_ids_equal": sum(
                a["prompt_token_ids"] == b["prompt_token_ids"] for a, b in pairs
            ),
            "image_grids_equal": sum(a["image_grid_thw"] == b["image_grid_thw"] for a, b in pairs),
            "full_token_ids_equal": sum(a["token_ids"] == b["token_ids"] for a, b in pairs),
            "prompt_token_counts": sorted({a["prompt_tokens"] for a, _ in pairs}),
            "full_visual_prefix_reused": {
                "replicated": sum(bool(a.get("full_visual_prefix_reused")) for a, _ in pairs),
                "data": sum(bool(b.get("full_visual_prefix_reused")) for _, b in pairs),
            },
        }
    result["comparison_status"] = "complete"
    return result


def before_eos(tokens: list[int]) -> list[int]:
    return tokens[: tokens.index(EOS_TOKEN_ID)] if EOS_TOKEN_ID in tokens else list(tokens)


def summarize_muir(reference_path: Path, candidate_path: Path) -> dict:
    reference = read_json(reference_path)
    candidate = read_json(candidate_path)
    left = {str(row["sample_id"]): row for row in reference["requests"]}
    right = {str(row["sample_id"]): row for row in candidate["requests"]}
    if (
        left.keys() != right.keys()
        or len(left) != len(reference["requests"])
        or len(right) != len(candidate["requests"])
    ):
        raise ValueError("MuirBench files must contain the same unique Sample IDs")
    rows = []
    for sample_id, a in left.items():
        b = right[sample_id]
        a_tokens, b_tokens = before_eos(a["token_ids"]), before_eos(b["token_ids"])
        a_choice = CHOICE_TOKENS.get(a_tokens[0]) if a_tokens else None
        b_choice = CHOICE_TOKENS.get(b_tokens[0]) if b_tokens else None
        if a["answer"] != b["answer"]:
            raise ValueError(f"MuirBench ground truth differs for Sample ID {sample_id}")
        rows.append(
            {
                "sample_id": sample_id,
                "ground_truth": a["answer"],
                "full_token_ids_exact": a["token_ids"] == b["token_ids"],
                "both_have_16_tokens": len(a["token_ids"]) == len(b["token_ids"]) == 16,
                "answer_token_ids_exact": a_tokens == b_tokens,
                "replicated_answer_token_ids": a_tokens,
                "data_answer_token_ids": b_tokens,
                "replicated_choice": a_choice,
                "data_choice": b_choice,
                "both_choices_parseable": a_choice is not None and b_choice is not None,
                "replicated_correct": a_choice == a["answer"],
                "data_correct": b_choice == b["answer"],
                "prompt_token_ids_exact": a["prompt_token_ids"] == b["prompt_token_ids"],
            }
        )
    old_parser = {}
    for label, report in (("replicated", reference), ("data", candidate)):
        requests = report["requests"]
        old_parser[label] = {
            "parsed": sum(row.get("parsed_answer") is not None for row in requests),
            "unparseable": sum(row.get("parsed_answer") is None for row in requests),
            "strict_correct": sum(bool(row.get("strict_correct")) for row in requests),
        }
    return {
        "sources": [source_name(reference_path), source_name(candidate_path)],
        "model": reference["model"],
        "options": {"replicated": reference["options"], "data": candidate["options"]},
        "selection": reference["selection"],
        "sampling": reference["sampling"],
        "scope": "paired 20-request implementation check, not official MuirBench accuracy",
        "parsing": {
            "eos_token_id": EOS_TOKEN_ID,
            "choice_token_mapping": CHOICE_TOKENS,
            "rule": (
                "cut at first EOS, map first answer token 32..36 to A..E; other tokens unparseable"
            ),
            "old_parser_issue": (
                "ignore_eos=True retained generated text after EOS in the old prediction; "
                "its strict-parser counts are diagnostic, not the corrected MCQ score."
            ),
            "old_parser_counts_preserved": old_parser,
        },
        "requests": len(rows),
        "both_have_16_token_count": sum(row["both_have_16_tokens"] for row in rows),
        "all_prompt_token_ids_equal": all(row["prompt_token_ids_exact"] for row in rows),
        "full_16_token_exact": sum(
            row["full_token_ids_exact"] and row["both_have_16_tokens"] for row in rows
        ),
        "answer_token_exact": sum(row["answer_token_ids_exact"] for row in rows),
        "valid_choice_pairs": sum(row["both_choices_parseable"] for row in rows),
        "valid_choice_agreement": sum(
            row["both_choices_parseable"] and row["replicated_choice"] == row["data_choice"]
            for row in rows
        ),
        "correct_answers": {
            mode: sum(row[f"{mode}_correct"] for row in rows) for mode in ("replicated", "data")
        },
        "non_choice_sample_ids": [
            row["sample_id"] for row in rows if not row["both_choices_parseable"]
        ],
        "full_token_mismatch_sample_ids": [
            row["sample_id"] for row in rows if not row["full_token_ids_exact"]
        ],
        "rows": rows,
    }


def summarize_replicas(path: Path) -> dict:
    raw = read_json(path)
    requests = [
        row
        for replica in raw["replicas"]
        for row in read_json(path.parent / Path(replica["raw_json"]).name)["requests"]
    ]
    phases = {}
    for name, phase in raw["phases"].items():
        phase_rows = [row for row in requests if row["phase"] == name]
        compact = summarize_requests(
            phase_rows,
            concurrent=name.startswith("concurrent_"),
        )
        compact["submitted_records"] = len(phase_rows)
        compact["all_records_finished"] = all(row["status"] == "finished" for row in phase_rows)
        for key in (
            "replica_first_request_start_skew_ms",
            "cross_replica_token_ids_exact_by_request_index",
            "full_visual_prefix_reused_requests",
        ):
            compact[key] = phase[key]
        phases[name] = compact
    return {
        "source": source_name(path),
        "status": raw["status"],
        "error": raw.get("error"),
        "resource_contract": raw["resource_contract"],
        "kv_cache": raw["kv_cache"],
        "timing_scope": raw["timing_scope"],
        "throughput_scope": raw["throughput_scope"],
        "phase_summaries_source": "recomputed from both raw replica request files",
        "scope": (
            "independent two-TP1-replica experiment; "
            "total KV/slots/offered requests need not match TP2"
        ),
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-replicated")
    parser.add_argument("--final-data")
    parser.add_argument("--replicas-summary")
    parser.add_argument("--vllm-pp2")
    args = parser.parse_args()
    if bool(args.final_replicated) != bool(args.final_data):
        parser.error("provide both --final-replicated and --final-data")
    raw_dir = args.raw_dir
    summary = {
        "schema_version": 2,
        "selection_policy": (
            "fixed named initial evidence; final TP2/replicas/PP2 require explicit CLI paths"
        ),
        "statistics_policy": (
            "Sequential phases report request latency and a scoped duration, not throughput; "
            "tok/s and req/s are emitted only for a single concurrent burst."
        ),
        "vision_features": summarize_features(raw_dir / "vision-features-9759.json"),
        "partition_diagnosis": summarize_diagnosis(raw_dir / "vision-partition-9763.json"),
        "initial_tp2_9764": summarize_tp2_pair(
            raw_dir / "tp2-replicated-9764.json", raw_dir / "tp2-data-9764.json"
        ),
        "final_tp2": None,
        "muirbench_9765": summarize_muir(
            raw_dir / "muir-replicated-9765.json", raw_dir / "muir-data-9765.json"
        ),
        "replicas": None,
    }
    if args.final_replicated:
        summary["final_tp2"] = summarize_tp2_pair(
            selected_path(raw_dir, args.final_replicated), selected_path(raw_dir, args.final_data)
        )
    if args.replicas_summary:
        summary["replicas"] = summarize_replicas(selected_path(raw_dir, args.replicas_summary))
    vllm_path = raw_dir / "vllm-tp2-9761.json"
    vllm_tp2 = read_json(vllm_path)
    summary["vllm_internal"] = {
        "scope": "vLLM eager BF16 TP2/PP2 direction only; not an absolute Prism ranking",
        "tp2": compact_report(vllm_tp2, vllm_path),
        "pp2": None,
    }
    if args.vllm_pp2:
        path = selected_path(raw_dir, args.vllm_pp2)
        pp2 = read_json(path)
        summary["vllm_internal"]["pp2"] = compact_report(pp2, path)
        summary["vllm_internal"]["same_input_fixture"] = vllm_tp2["input"] == pp2["input"]
        summary["vllm_internal"]["same_sampling"] = vllm_tp2["sampling"] == pp2["sampling"]
        if pp2.get("status") == vllm_tp2.get("status") == "complete":
            tp2_phases = summary["vllm_internal"]["tp2"]["phase_summaries"]
            pp2_phases = summary["vllm_internal"]["pp2"]["phase_summaries"]
            summary["vllm_internal"]["pp2_vs_tp2_change_percent"] = {
                phase: percent_changes(values, pp2_phases[phase])
                for phase, values in tp2_phases.items()
                if phase in pp2_phases
            }
    # Failure inventory is not a selector: no successful run is promoted here.
    summary["retained_failures"] = []
    for path in sorted(raw_dir.glob("*.json")):
        record = read_json(path)
        if record.get("status") == "failed":
            summary["retained_failures"].append(
                {
                    "source": source_name(path),
                    "error": record.get("error"),
                    "completed_request_records": Counter(
                        row.get("status") for row in record.get("requests", [])
                    ).get("finished", 0),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
