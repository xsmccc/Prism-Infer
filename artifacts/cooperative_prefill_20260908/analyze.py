"""Derive the interleaved-Prefill report from retained request timestamps."""

from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
PAIRS = ((9799, 9798), (9800, 9801))  # (off, on), opposite execution orders
METRICS = (
    ("decode_with_cold_image", "max_itl_ms"),
    ("decode_with_cold_image", "p95_itl_ms"),
    ("decode_with_cold_image", "tpot_ms"),
    ("decode_with_cold_image", "e2e_ms"),
    ("injected_cold_image", "ttft_ms"),
    ("injected_cold_image", "e2e_ms"),
)


def read(job):
    return json.loads((RAW / f"http-cooperative-{job}.json").read_text(encoding="utf-8"))


def output_differences(reference, requests):
    differences = []
    for index, (a, b) in enumerate(zip(reference, requests, strict=True)):
        if a["token_ids"] == b["token_ids"]:
            continue
        first = next(
            i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"], strict=True)) if x != y
        )
        differences.append(
            {
                "request_index": index,
                "phase": a["phase"],
                "first_different_token_index": first,
                "reference_token": a["token_ids"][first],
                "run_token": b["token_ids"][first],
                "eos_before_difference": any(t in (151643, 151645) for t in a["token_ids"][:first]),
            }
        )
    return differences


def summarize():
    runs = {job: read(job) for pair in PAIRS for job in pair}
    reference = runs[9799]["requests"]
    comparison = []
    for phase, metric in METRICS:
        values = [[runs[j]["phase_summaries"][phase][metric] for j in pair] for pair in PAIRS]
        changes = [100 * (b / a - 1) for a, b in values]
        comparison.append(
            {
                "phase": phase,
                "metric": metric,
                "off_run_values_ms": [v[0] for v in values],
                "on_run_values_ms": [v[1] for v in values],
                "off_median_ms": statistics.median(v[0] for v in values),
                "on_median_ms": statistics.median(v[1] for v in values),
                "paired_change_percent": changes,
            }
        )
    return {
        "source_base": "ae5b921",
        "scope": "TP1 RTX 5090, same native HTTP fixture; not a vLLM ranking or quality benchmark",
        "pairs_off_on": PAIRS,
        "run_status": {job: run["status"] for job, run in runs.items()},
        "request_count_per_run": len(reference),
        "generated_tokens_per_run": sum(len(r["token_ids"]) for r in reference),
        "agreement": {
            job: {
                "same_inputs": len(run["requests"]) == len(reference)
                and all(
                    all(
                        a[key] == b[key] for key in ("phase", "prompt", "media_group", "max_tokens")
                    )
                    for a, b in zip(reference, run["requests"], strict=True)
                ),
                "matching_full_outputs": sum(
                    a["token_ids"] == b["token_ids"]
                    for a, b in zip(reference, run["requests"], strict=True)
                ),
                "stream_matches_final": all(r["stream_matches_final_ids"] for r in run["requests"]),
                "differences_from_9799": output_differences(reference, run["requests"]),
            }
            for job, run in runs.items()
        },
        "aggregation": "Each run: median over 3 requests. Table: median of the 2 run medians.",
        "itl_note": (
            "max and p95 are per-request token gaps, then request medians; "
            "neither is population p99"
        ),
        "comparison": comparison,
    }


def trace():
    with sqlite3.connect(RAW / "prefill-9802.sqlite") as db:
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
            r["name"] = r["text"] or strings.get(r["textId"], "NVTX")
        quanta = [
            r
            for r in regions
            if r["name"]
            in (
                "runner.prefill.vision_block_quantum",
                "runner.prefill.language_layer_quantum",
            )
        ]
        left, right = quanta[0]["start"] - 50_000_000, quanta[-1]["end"] + 50_000_000
        selected = [r for r in regions if r["end"] > left and r["start"] < right]
        kernels = [
            dict(r)
            for r in db.execute(
                "SELECT start,end,deviceId,streamId,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE end>? AND start<? ORDER BY start",
                (left, right),
            )
        ]
    events = []
    for r in selected:
        events.append(
            {
                "ph": "X",
                "cat": "CPU NVTX",
                "pid": 1,
                "tid": str(r["globalTid"]),
                "name": r["name"],
                "ts": (max(left, r["start"]) - left) / 1000,
                "dur": (min(right, r["end"]) - max(left, r["start"])) / 1000,
            }
        )
    for r in kernels:
        events.append(
            {
                "ph": "X",
                "cat": "GPU kernel",
                "pid": r["deviceId"] + 100,
                "tid": r["streamId"],
                "name": strings[r["shortName"]],
                "ts": (max(left, r["start"]) - left) / 1000,
                "dur": (min(right, r["end"]) - max(left, r["start"])) / 1000,
            }
        )
    replays = [r for r in selected if r["name"] == "runner.cudagraph.replay"]
    report = {
        "source": "raw/prefill-9802.nsys-rep",
        "vision_quanta": sum(r["name"].endswith("vision_block_quantum") for r in quanta),
        "language_quanta": sum(r["name"].endswith("language_layer_quantum") for r in quanta),
        "quantum_gaps_with_decode_replay": sum(
            any(a["end"] <= r["start"] < b["start"] for r in replays)
            for a, b in zip(quanta, quanta[1:], strict=False)
        ),
        "total_quantum_gaps": len(quanta) - 1,
        "note": (
            "Host NVTX and GPU kernels are separate tracks; "
            "interleaving is not simultaneous GPU execution."
        ),
    }
    (ROOT / "trace.json").write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    (ROOT / "trace_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = summarize()
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if (RAW / "prefill-9802.sqlite").exists():
        print(json.dumps(trace(), indent=2))
