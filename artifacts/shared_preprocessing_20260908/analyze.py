"""Summarize the paired HTTP requests and crop the cold-preparation Nsight timeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def read(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def union_ns(intervals):
    if not intervals:
        return 0
    start, end = sorted(intervals)[0]
    total = 0
    for left, right in sorted(intervals)[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def trace():
    with sqlite3.connect(RAW / "preprocessing-9791.sqlite") as database:
        database.row_factory = sqlite3.Row
        strings = dict(database.execute("SELECT id,value FROM StringIds"))
        regions = [
            dict(row)
            for row in database.execute(
                "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS "
                "WHERE end IS NOT NULL AND end>start ORDER BY start"
            )
        ]
        for region in regions:
            region["name"] = region["text"] or strings.get(region["textId"], "NVTX")
        preparation = [r for r in regions if r["name"] == "preprocess.image_processor"]
        # The trace run has warmup then one content-distinct cold injection.
        cold = preparation[1]
        left = cold["start"] - 100_000_000
        right = cold["end"] + 350_000_000
        kernels = [
            dict(row)
            for row in database.execute(
                "SELECT start,end,deviceId,streamId,shortName,demangledName "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>? AND start<? ORDER BY start",
                (left, right),
            )
        ]
    selected = [r for r in regions if r["end"] > left and r["start"] < right]
    tids = {tid: index for index, tid in enumerate(sorted({r["globalTid"] for r in selected}))}
    events = [{"ph": "M", "pid": 1, "tid": 0, "name": "process_name", "args": {"name": "CPU NVTX"}}]
    for tid, index in tids.items():
        label = "Image preprocessing worker" if tid == cold["globalTid"] else f"CPU thread {tid}"
        events.append(
            {"ph": "M", "pid": 1, "tid": index, "name": "thread_name", "args": {"name": label}}
        )
    for device in sorted({r["deviceId"] for r in kernels}):
        events.append(
            {
                "ph": "M",
                "pid": device + 100,
                "tid": 0,
                "name": "process_name",
                "args": {"name": f"GPU {device}"},
            }
        )
    for region in selected:
        start, end = max(left, region["start"]), min(right, region["end"])
        events.append(
            {
                "ph": "X",
                "cat": "CPU",
                "pid": 1,
                "tid": tids[region["globalTid"]],
                "name": region["name"],
                "ts": (start - left) / 1000,
                "dur": (end - start) / 1000,
            }
        )
    overlap = []
    decode_kernels = 0
    for kernel in kernels:
        start, end = max(left, kernel["start"]), min(right, kernel["end"])
        name = strings[kernel["shortName"]]
        events.append(
            {
                "ph": "X",
                "cat": "GPU",
                "pid": kernel["deviceId"] + 100,
                "tid": kernel["streamId"],
                "name": name,
                "ts": (start - left) / 1000,
                "dur": (end - start) / 1000,
            }
        )
        a, b = max(cold["start"], kernel["start"]), min(cold["end"], kernel["end"])
        if b > a:
            overlap.append((a, b))
            decode_kernels += int("paged_decode" in name)
    replay = [
        r
        for r in selected
        if r["name"] == "runner.cudagraph.replay"
        and r["start"] < cold["end"]
        and r["end"] > cold["start"]
    ]
    (ROOT / "trace.json").write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}), encoding="utf-8"
    )
    return {
        "source": "raw/preprocessing-9791.nsys-rep",
        "selected_preprocessing_call": 1,
        "scope": "single-GPU trace; kernel overlap is not an end-to-end speedup",
        "cpu_image_processor_ms": (cold["end"] - cold["start"]) / 1e6,
        "gpu_kernel_activity_overlapping_cpu_ms": union_ns(overlap) / 1e6,
        "paged_decode_kernels_overlapping_cpu": decode_kernels,
        "decode_replay_calls_during_cpu": len(replay),
        "preprocessing_thread": str(cold["globalTid"]),
        "decode_replay_threads": sorted({str(r["globalTid"]) for r in replay}),
        "chrome_trace": "trace.json",
    }


def main():
    before = read("http-baseline-9788.json")
    after = read("http-candidate-9790.json")
    pairs = list(zip(before["requests"], after["requests"], strict=True))
    summaries = {}
    for phase, original in before["phase_summaries"].items():
        if phase == "warmup":
            continue
        updated = after["phase_summaries"][phase]
        summaries[phase] = {
            "before": original,
            "after": updated,
            "latency_reduction_percent": {
                key: 100 * (1 - updated[key] / value) for key, value in original.items()
            },
        }
    report = {
        "baseline": "8145043",
        "scope": "same HTTP workload, TP1, RTX 5090, eight 448x448 fixture images",
        "model_options": read("server-candidate-9790.json")["options"],
        "model_options_equal": read("server-candidate-9790.json")["options"]
        == read("server-baseline-9788.json")["options"],
        "requests": len(pairs),
        "same_inputs": all(
            all(a[k] == b[k] for k in ("phase", "prompt", "media_group", "max_tokens"))
            for a, b in pairs
        ),
        "same_full_token_sequences": sum(a["token_ids"] == b["token_ids"] for a, b in pairs),
        "total_generated_tokens_per_run": sum(len(a["token_ids"]) for a, _ in pairs),
        "all_streams_match_final_ids": all(
            a["stream_matches_final_ids"] and b["stream_matches_final_ids"] for a, b in pairs
        ),
        "all_cold_requests_injected_during_decode": all(
            not r["long_already_finished_at_submit"]
            for d in (before, after)
            for r in d["requests"]
            if r["phase"] == "injected_cold_image"
        ),
        "metric_note": "max_itl_ms is the median of each request's maximum, not population p99",
        "phase_comparison": summaries,
        "media_cache": read("server-candidate-9790.json")["media_preprocess_cache"],
        "trace": trace(),
    }
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["trace"], indent=2))


if __name__ == "__main__":
    main()
