"""Export this Vision-DP Nsight SQLite's two request windows to Chrome Trace.

Includes completed NVTX ranges and GPU kernels, optionally CUDA memcpy records.
This is intentionally limited to the schema exported for vision-dp-9769.sqlite.
Chrome timestamps are microseconds relative to trace_cold's NVTX start; the two
phases retain their original spacing and also expose phase-relative timestamps.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

PHASE_NAMES = ("trace_cold", "trace_full_prefix_hit")


def overlap(start: int, end: int, phase: dict) -> tuple[int, int] | None:
    clipped_start, clipped_end = max(start, phase["start"]), min(end, phase["end"])
    return (clipped_start, clipped_end) if clipped_end > clipped_start else None


def union_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def metadata(name: str, pid: int, tid: int, args: dict) -> dict:
    return {"name": name, "ph": "M", "pid": pid, "tid": tid, "args": args}


def duration_event(
    name: str,
    category: str,
    pid: int,
    tid: int,
    start: int,
    end: int,
    origin: int,
    phase: dict,
    args: dict,
) -> dict:
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": (start - origin) / 1000,
        "dur": (end - start) / 1000,
        "args": {
            "phase": phase["name"],
            "phase_relative_start_us": (start - phase["start"]) / 1000,
            **args,
        },
    }


def convert(sqlite_path: Path, include_memcpy: bool) -> tuple[dict, dict]:
    connection = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        nvtx = [
            dict(row) for row in connection.execute(
                "SELECT start,end,globalTid,endGlobalTid,text,textId FROM NVTX_EVENTS "
                "WHERE end IS NOT NULL AND end > start ORDER BY start"
            )
        ]
        for row in nvtx:
            row["name"] = row["text"] or strings.get(row["textId"], "unnamed NVTX")
        phases = []
        for name in PHASE_NAMES:
            matches = [row for row in nvtx if row["name"] == name]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one completed {name} range, got {len(matches)}")
            phases.append(matches[0])
        origin = min(phase["start"] for phase in phases)
        end = max(phase["end"] for phase in phases)
        kernels = [
            dict(row) for row in connection.execute(
                "SELECT start,end,deviceId,contextId,streamId,globalPid,correlationId,"
                "shortName,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE end > ? AND start < ? ORDER BY start", (origin, end)
            )
        ]
        copies = []
        if include_memcpy:
            copies = [
                dict(row) for row in connection.execute(
                    "SELECT start,end,deviceId,contextId,streamId,globalPid,correlationId,"
                    "bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                    "WHERE end > ? AND start < ? ORDER BY start", (origin, end)
                )
            ]
        copy_names = dict(connection.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER"))
        gpu_info = {
            row["id"]: dict(row) for row in connection.execute(
                "SELECT id,name,uuid,busLocation FROM TARGET_INFO_GPU"
            )
        }
    finally:
        connection.close()

    active_nvtx = [
        row for row in nvtx if any(overlap(row["start"], row["end"], phase) for phase in phases)
    ]
    cpu_tids = {global_tid: index for index, global_tid in enumerate(sorted(
        {row["globalTid"] for row in active_nvtx}
    ))}
    gpu_tracks = {}
    for row in [*kernels, *copies]:
        key = row["deviceId"], row["globalPid"], row["contextId"], row["streamId"]
        if key not in gpu_tracks:
            gpu_tracks[key] = len(gpu_tracks)

    events = [
        metadata("process_name", 1, 0, {"name": "Request phases"}),
        metadata("thread_name", 1, 0, {"name": "Cold then full-prefix hit"}),
        metadata("process_name", 2, 0, {"name": "CPU NVTX (globalTid tracks)"}),
    ]
    for global_tid, tid in cpu_tids.items():
        events.append(metadata("thread_name", 2, tid, {"name": f"globalTid {global_tid}"}))
    for device in sorted({key[0] for key in gpu_tracks}):
        name = gpu_info.get(device, {}).get("name", "CUDA GPU")
        events.append(metadata("process_name", 100 + device, 0, {"name": f"GPU {device}: {name}"}))
    for (device, global_pid, context, stream), tid in gpu_tracks.items():
        events.append(metadata("thread_name", 100 + device, tid, {
            "name": f"stream {stream} / context {context} / globalPid {global_pid}",
        }))

    summaries = {}
    for phase in phases:
        events.append(duration_event(
            phase["name"], "request phase", 1, 0, phase["start"], phase["end"], origin, phase, {}
        ))
        vision_counts: Counter = Counter()
        vision_by_thread: dict[str, Counter] = {}
        nvtx_count = 0
        for row in active_nvtx:
            interval = overlap(row["start"], row["end"], phase)
            if interval is None:
                continue
            nvtx_count += 1
            if "model.vision" in row["name"]:
                vision_counts[row["name"]] += 1
                thread = str(row["globalTid"])
                vision_by_thread.setdefault(thread, Counter())[row["name"]] += 1
            events.append(duration_event(
                row["name"], "NVTX", 2, cpu_tids[row["globalTid"]], *interval, origin, phase,
                {
                    "globalTid": str(row["globalTid"]),
                    "endGlobalTid": str(row["endGlobalTid"]) if row["endGlobalTid"] else None,
                    "clipped_to_phase": interval != (row["start"], row["end"]),
                },
            ))

        device_intervals: dict[int, list[tuple[int, int]]] = {}
        kernel_totals: dict[str, Counter] = {}
        for row in kernels:
            interval = overlap(row["start"], row["end"], phase)
            if interval is None:
                continue
            device = row["deviceId"]
            device_intervals.setdefault(device, []).append(interval)
            name = strings.get(row["shortName"], strings.get(row["demangledName"], "CUDA kernel"))
            kernel_totals.setdefault(name, Counter())["count"] += 1
            kernel_totals[name]["duration_ns"] += interval[1] - interval[0]
            track = (device, row["globalPid"], row["contextId"], row["streamId"])
            events.append(duration_event(
                name, "CUDA kernel", 100 + device, gpu_tracks[track], *interval, origin, phase,
                {
                    "correlationId": row["correlationId"],
                    "demangledNameStringId": row["demangledName"],
                    "clipped_to_phase": interval != (row["start"], row["end"]),
                },
            ))
        phase_copies = []
        for row in copies:
            interval = overlap(row["start"], row["end"], phase)
            if interval is None:
                continue
            phase_copies.append(row)
            track = (row["deviceId"], row["globalPid"], row["contextId"], row["streamId"])
            events.append(duration_event(
                f"Memcpy {copy_names.get(row['copyKind'], row['copyKind'])}", "CUDA memcpy",
                100 + row["deviceId"], gpu_tracks[track], *interval, origin, phase,
                {"bytes": row["bytes"], "correlationId": row["correlationId"]},
            ))
        all_intervals = [interval for values in device_intervals.values() for interval in values]
        summaries[phase["name"]] = {
            "source_start_ns": phase["start"],
            "trace_offset_ms": (phase["start"] - origin) / 1e6,
            "nvtx_request_window_ms": (phase["end"] - phase["start"]) / 1e6,
            "nvtx_ranges": nvtx_count,
            "vision_nvtx_counts": dict(vision_counts),
            "vision_by_global_tid": {key: dict(value) for key, value in vision_by_thread.items()},
            "vision_encoder_threads": len(vision_by_thread),
            "vision_encode_calls": vision_counts["model.vision.data_parallel.encode"],
            "vision_gather_calls": vision_counts["model.vision.data_parallel.gather"],
            "gpu_kernel_records": len(all_intervals),
            "all_devices_kernel_duration_sum_ms": sum(b - a for a, b in all_intervals) / 1e6,
            "all_devices_kernel_timeline_span_ms": (
                (max(b for _, b in all_intervals) - min(a for a, _ in all_intervals)) / 1e6
                if all_intervals else 0
            ),
            "per_device": {
                str(device): {
                    "kernel_records": len(values),
                    "kernel_duration_sum_ms": sum(b - a for a, b in values) / 1e6,
                    "kernel_activity_union_ms": union_duration(values) / 1e6,
                    "kernel_timeline_span_ms": (
                        max(b for _, b in values) - min(a for a, _ in values)
                    ) / 1e6,
                }
                for device, values in device_intervals.items()
            },
            "top_kernels_by_accumulated_duration": [
                {
                    "name": name, "count": values["count"],
                    "duration_sum_ms": values["duration_ns"] / 1e6,
                }
                for name, values in sorted(
                    kernel_totals.items(), key=lambda item: item[1]["duration_ns"], reverse=True
                )[:10]
            ],
            "memcpy_records": len(phase_copies),
            "memcpy_bytes_full_records": sum(row["bytes"] for row in phase_copies),
        }

    summary = {
        "source_sqlite": sqlite_path.name,
        "trace_origin_ns": origin,
        "time_definition": "Chrome ts/dur are microseconds; phases retain actual relative spacing",
        "cpu_mapping": "opaque globalTid mapped to presentation tracks; no PID/TID bit decoding",
        "gpu_mapping": "deviceId process track, distinct globalPid/contextId/streamId lanes",
        "timing_caution": (
            "NVTX is CPU range timing. Kernel sums count overlapping work across streams/GPUs; "
            "timeline spans include gaps. Neither can be added to the NVTX request duration, "
            "nor substituted for an unprofiled end-to-end benchmark. "
            "GPU activity union is per device."
        ),
        "gpu_devices": gpu_info,
        "included_memcpy": include_memcpy,
        "phases": summaries,
        "chrome_event_records": len(events),
    }
    events.sort(key=lambda event: (
        event["ph"] != "M", event.get("ts", 0), -event.get("dur", 0)
    ))
    trace = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "otherData": {
            "source": sqlite_path.name,
            "timestamp_origin_ns": origin,
            "presentation_ids_are_not_os_ids": True,
            "phase_names": list(PHASE_NAMES),
            "timing_caution": summary["timing_caution"],
        },
    }
    return trace, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--include-memcpy", action="store_true")
    args = parser.parse_args()
    trace, summary = convert(args.sqlite, args.include_memcpy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, separators=(",", ":")) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "trace": str(args.output), "summary": str(args.summary),
        "events": summary["chrome_event_records"],
        "vision_counts": {
            name: {key: phase[key] for key in ("vision_encode_calls", "vision_gather_calls")}
            for name, phase in summary["phases"].items()
        },
    }))


if __name__ == "__main__":
    main()
