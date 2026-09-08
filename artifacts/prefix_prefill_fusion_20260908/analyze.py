"""Summarize the paired HTTP run and attribute hot Prefill CUDA launches."""

from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
PAIR_JOB = 9816


def read(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def supporting_measurements():
    ablation = read("ablate-prefill-fused-9819.json")
    profile_counts = {}
    for source in ("cooperative", "prefill-fused"):
        records = read(f"cpu-{source}-9817.json")
        profile_counts[source] = [
            {
                "query_tokens": r["scheduled_tokens"],
                "profiled_step_ms": r["step_ms"],
                "device_mode_calls": sum(
                    f["calls"]
                    for f in r["functions"]
                    if f["name"] == "__torch_function__"
                    and f["file"].endswith("/torch/utils/_device.py")
                ),
            }
            for r in records[1:]
        ]
    return {
        "device_mode_cpu_profiles": profile_counts,
        "fusion_ablation": {
            "job": 9819,
            "scope": "same process, device mode already repaired; no HTTP or preprocessing",
            "prefill_step_median_ms": {
                str(fused): statistics.median(
                    r["prefill_step_ms"]
                    for r in ablation
                    if r["fused"] == fused and not r["warmup"]
                )
                for fused in (False, True)
            },
            "all_outputs_equal": all(r["token_ids"] == ablation[0]["token_ids"] for r in ablation),
        },
    }


def summarize(pair_job=PAIR_JOB):
    cells = ("baseline-1", "candidate-1", "candidate-2", "baseline-2")
    runs = {cell: read(f"http-fusion-{cell}-{pair_job}.json") for cell in cells}
    ref = runs["baseline-1"]["requests"]
    metrics = {}
    for phase in runs["baseline-1"]["phase_summaries"]:
        if phase == "warmup":
            continue
        metrics[phase] = {}
        for key in ("ttft_ms", "e2e_ms", "tpot_ms", "max_itl_ms", "p95_itl_ms"):
            base = [runs[c]["phase_summaries"][phase][key] for c in ("baseline-1", "baseline-2")]
            new = [runs[c]["phase_summaries"][phase][key] for c in ("candidate-1", "candidate-2")]
            metrics[phase][key] = {
                "baseline_runs": base,
                "candidate_runs": new,
                "baseline_median": statistics.median(base),
                "candidate_median": statistics.median(new),
                "paired_change_percent": [
                    100 * (b / a - 1) for a, b in zip(base, new, strict=True)
                ],
            }
    outputs = {}
    for cell, run in runs.items():
        differences = []
        for i, (a, b) in enumerate(zip(ref, run["requests"], strict=True)):
            if a["token_ids"] != b["token_ids"]:
                first = next(
                    j
                    for j, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"], strict=True))
                    if x != y
                )
                differences.append(
                    {"request_index": i, "phase": a["phase"], "first_token_index": first}
                )
        outputs[cell] = {
            "status": run["status"],
            "requests": len(run["requests"]),
            "generated_tokens": sum(len(r["token_ids"]) for r in run["requests"]),
            "same_inputs": all(
                all(a[k] == b[k] for k in ("phase", "prompt", "media_group", "max_tokens"))
                for a, b in zip(ref, run["requests"], strict=True)
            ),
            "streams_match_final": all(r["stream_matches_final_ids"] for r in run["requests"]),
            "differences_from_baseline_1": differences,
        }
    return {
        "parent": "cc77b08",
        "pair_job": pair_job,
        "execution_order": cells,
        "scope": "TP1 RTX 5090 native HTTP fixture; cooperative/chunked Prefill off",
        "aggregation": "Per-run request medians, then median of two runs. Warmup excluded.",
        "outputs": outputs,
        "metrics": metrics,
    }


def profile(path, *, write_trace=False):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        regions = [
            dict(r)
            for r in db.execute(
                "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS "
                "WHERE end>start ORDER BY start"
            )
        ]
        for r in regions:
            r["name"] = r["text"] or strings.get(r["textId"], "")
        runtime = {
            r["correlationId"]: dict(r)
            for r in db.execute(
                "SELECT start,end,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        }
        kernels = [dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    sdpa = [r for r in regions if r["name"] == "attention.prefill.paged_vectorized_sdpa"]
    parents = [
        r
        for r in regions
        if r["name"] == "engine.model_runner"
        and any(r["start"] <= c["start"] and c["end"] <= r["end"] for c in sdpa)
    ]
    summaries = []
    events = []
    for index, parent in enumerate(parents):
        selected = []
        for k in kernels:
            call = runtime.get(k["correlationId"])
            if (
                call
                and call["globalTid"] == parent["globalTid"]
                and (parent["start"] <= call["start"] < parent["end"])
            ):
                selected.append(k)
        summaries.append(
            {
                "hot_prefill_index": index,
                "host_ms": (parent["end"] - parent["start"]) / 1e6,
                "gpu_kernel_ms": sum(k["end"] - k["start"] for k in selected) / 1e6,
                "gpu_span_ms": (max(k["end"] for k in selected) - min(k["start"] for k in selected))
                / 1e6,
                "kernel_count": len(selected),
                "fused_gather_calls": sum(
                    strings[k["shortName"]] == "_gather_paged_kv_kernel" for k in selected
                ),
                "fused_qk_calls": sum(
                    strings[k["shortName"]] == "_prefill_qk_normalize_mrope_kernel"
                    for k in selected
                ),
            }
        )
        if write_trace and index == 1:
            left = parent["start"]
            for r in regions:
                if r["globalTid"] == parent["globalTid"] and left <= r["start"] < parent["end"]:
                    events.append(
                        {
                            "ph": "X",
                            "cat": "CPU NVTX",
                            "pid": 1,
                            "tid": 1,
                            "name": r["name"],
                            "ts": (r["start"] - left) / 1000,
                            "dur": (r["end"] - r["start"]) / 1000,
                        }
                    )
            for k in selected:
                events.append(
                    {
                        "ph": "X",
                        "cat": "GPU",
                        "pid": 2,
                        "tid": k["streamId"],
                        "name": strings[k["demangledName"]],
                        "ts": (k["start"] - left) / 1000,
                        "dur": (k["end"] - k["start"]) / 1000,
                    }
                )
    if write_trace:
        (ROOT / "trace.json").write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    result = summarize()
    result.update(supporting_measurements())
    result["fusions_only_exploration"] = {
        "pair_job": 9813,
        "hot_ttft": summarize(9813)["metrics"]["hot_changed_question"]["ttft_ms"],
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if (RAW / "fusion-9818.sqlite").exists():
        profiles = {"candidate": profile(RAW / "fusion-9818.sqlite", write_trace=True)}
        if (RAW / "fusion-9814.sqlite").exists():
            profiles["fusions_only"] = profile(RAW / "fusion-9814.sqlite")
        old = ROOT.parent / "cooperative_prefill_20260908/raw/prefill-9802.sqlite"
        if old.exists():
            profiles["previous_code"] = profile(old)
        (ROOT / "trace_summary.json").write_text(
            json.dumps(profiles, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(profiles, indent=2))
