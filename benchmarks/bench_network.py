#!/usr/bin/env python3
"""Measure Prism-Infer through its real HTTP/SSE serving boundary."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_online import (  # noqa: E402
    _DeviceProcessMemorySampler,
    _arrival_offsets,
    _canonical_sha256,
    _h3_conformance,
    _h3_payload_schedule,
    _load_class_slos,
)
from benchmarks.harness import collect_git_metadata, collect_gpu_metadata  # noqa: E402
from prism_infer.analysis.benchmark_schema import load_workload_manifest  # noqa: E402
from prism_infer.analysis.online_serving import summarize_distribution  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "workloads" / "p9_headline.json"
TERMINAL_EVENTS = frozenset({"done", "error"})


def _image_data_url(image: Image.Image) -> str:
    """Encode one materialized image without changing its RGB pixels."""

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _network_payload(payload: dict[str, Any]) -> dict[str, object]:
    """Convert an existing frozen workload payload into the HTTP protocol."""

    request_type = str(payload.get("type", "text"))
    body: dict[str, object] = {
        "prompt": payload["prompt"],
        "modality": "text" if request_type == "text" else request_type.rstrip("s"),
    }
    if request_type == "image":
        body["media"] = _image_data_url(payload["image"])
    elif request_type == "images":
        body["media"] = [_image_data_url(image) for image in payload["images"]]
    elif request_type == "video":
        body["media"] = [_image_data_url(frame) for frame in payload["video"]]
    elif request_type != "text":
        raise ValueError(f"unsupported network workload request type: {request_type!r}")
    return body


async def _wait_until(target: float) -> None:
    wait_seconds = target - time.perf_counter()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


async def _run_one(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    body_template: dict[str, object],
    request_id: str,
    request_class: str,
    arrival_offset_s: float,
    started: float,
    max_tokens: int,
) -> dict[str, object]:
    """Send one scheduled request and measure events at the socket boundary."""

    intended_arrival = started + arrival_offset_s
    await _wait_until(intended_arrival)
    observed_submit = time.perf_counter()
    body = {
        **body_template,
        "request_id": request_id,
        "sampling": {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    token_ids: list[int] = []
    token_arrivals: list[float] = []
    finish_reason: str | None = None
    terminal_event: str | None = None
    error: str | None = None
    async with session.post(endpoint, json=body) as response:
        if response.status != 200:
            detail = await response.text()
            raise RuntimeError(
                f"network request {request_id} failed with HTTP {response.status}: {detail}"
            )
        event_name: str | None = None
        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line:
                event_name = None
                continue
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
                continue
            if not line.startswith("data:"):
                continue
            event = json.loads(line.removeprefix("data:").strip())
            observed = time.perf_counter()
            kind = str(event.get("event", event_name))
            if kind == "token":
                token_ids.append(int(event["token_id"]))
                token_arrivals.append(observed)
            elif kind in TERMINAL_EVENTS:
                terminal_event = kind
                finish_reason = event.get("finish_reason")
                error = event.get("error")
                break
    finished = time.perf_counter()
    if terminal_event != "done" or not token_arrivals:
        raise RuntimeError(
            f"network request {request_id} did not complete: "
            f"event={terminal_event!r} error={error!r}"
        )
    first_token = token_arrivals[0]
    return {
        "request_id": request_id,
        "request_class": request_class,
        "arrival_offset_s": arrival_offset_s,
        "controller_submit_delay_ms": (observed_submit - intended_arrival) * 1000.0,
        "ttft_ms": (first_token - intended_arrival) * 1000.0,
        "tpot_ms": (
            0.0
            if len(token_ids) <= 1
            else (finished - first_token) * 1000.0 / (len(token_ids) - 1)
        ),
        "latency_ms": (finished - intended_arrival) * 1000.0,
        "output_tokens": len(token_ids),
        "token_ids": token_ids,
        "finish_reason": finish_reason,
    }


async def _run_arrivals(
    *,
    endpoint: str,
    payloads: list[dict[str, object]],
    request_classes: list[str],
    offsets_s: list[float],
    max_tokens: int,
    key_prefix: str,
) -> dict[str, object]:
    if not (len(payloads) == len(request_classes) == len(offsets_s)):
        raise ValueError("payload, class and arrival schedules must have equal lengths")
    timeout = aiohttp.ClientTimeout(total=None, connect=30.0)
    connector = aiohttp.TCPConnector(limit=0)
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _run_one(
                    session,
                    endpoint=endpoint,
                    body_template=payload,
                    request_id=f"{key_prefix}-{index:05d}",
                    request_class=request_classes[index],
                    arrival_offset_s=offsets_s[index],
                    started=started,
                    max_tokens=max_tokens,
                )
            )
            for index, payload in enumerate(payloads)
        ]
        requests = await asyncio.gather(*tasks)
    finished = time.perf_counter()
    return {
        "started_perf_counter_s": started,
        "finished_perf_counter_s": finished,
        "duration_s": finished - started,
        "requests": requests,
    }


def _summary(
    run: dict[str, object],
    class_slos: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Aggregate network-visible latency, throughput and class-aware goodput."""

    duration_s = float(run["duration_s"])
    requests = list(run["requests"])
    completed = [
        record
        for record in requests
        if record.get("finish_reason") in {"eos", "length"}
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in completed:
        grouped.setdefault(str(record["request_class"]), []).append(record)

    good_requests = 0
    good_tokens = 0
    by_class: dict[str, object] = {}
    for case_id, records in sorted(grouped.items()):
        slo = class_slos.get(case_id)
        good = []
        if slo is not None:
            good = [
                record
                for record in records
                if float(record["ttft_ms"]) <= slo["ttft_ms"]
                and float(record["tpot_ms"]) <= slo["tpot_ms"]
            ]
        good_requests += len(good)
        good_tokens += sum(int(record["output_tokens"]) for record in good)
        by_class[case_id] = {
            "slo": slo,
            "counts": {
                "completed": len(records),
                "good": None if slo is None else len(good),
            },
            "latency_ms": {
                metric: summarize_distribution(
                    [float(record[f"{metric}_ms"]) for record in records]
                )
                for metric in ("ttft", "tpot", "latency")
            },
            "throughput": {
                "requests_per_s": len(records) / duration_s,
                "output_tokens_per_s": (
                    sum(int(record["output_tokens"]) for record in records) / duration_s
                ),
            },
            "goodput": (
                None
                if slo is None
                else {
                    "requests_per_s": len(good) / duration_s,
                    "output_tokens_per_s": (
                        sum(int(record["output_tokens"]) for record in good) / duration_s
                    ),
                    "fraction_of_completed": len(good) / len(records),
                }
            ),
        }

    return {
        "counts": {
            "submitted": len(requests),
            "completed": len(completed),
            "finish_reasons": dict(Counter(record["finish_reason"] for record in requests)),
        },
        "latency_ms": {
            metric: summarize_distribution(
                [float(record[f"{metric}_ms"]) for record in completed]
            )
            for metric in ("controller_submit_delay", "ttft", "tpot", "latency")
        },
        "throughput": {
            "requests_per_s": len(completed) / duration_s,
            "output_tokens_per_s": (
                sum(int(record["output_tokens"]) for record in completed) / duration_s
            ),
        },
        "class_aware": {
            "headline_unit": "output_tokens_per_second_meeting_both_slos",
            "slo_available": bool(class_slos),
            "by_class": by_class,
            "aggregate_goodput": (
                None
                if not class_slos
                else {
                    "requests_per_s": good_requests / duration_s,
                    "output_tokens_per_s": good_tokens / duration_s,
                }
            ),
        },
    }


async def _health(endpoint: str) -> dict[str, object]:
    health_url = endpoint.rsplit("/", 2)[0] + "/health"
    timeout = aiohttp.ClientTimeout(total=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(health_url) as response:
            payload = await response.json()
            if response.status != 200 or payload.get("status") != "ok":
                raise RuntimeError(f"Prism server is not healthy: HTTP {response.status} {payload}")
            return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18080/v1/generate",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--h3-profile",
        choices=("primary", "conditional_video"),
        default="conditional_video",
    )
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument(
        "--arrival-process",
        choices=("constant", "poisson", "burst"),
        default="poisson",
    )
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--class-slo-file")
    parser.add_argument("--server-config")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.requests <= 0 or args.warmup_requests < 0:
        parser.error("--requests must be positive and --warmup-requests must be non-negative")

    manifest = load_workload_manifest(args.manifest)
    payloads, request_classes, h3_contract = _h3_payload_schedule(
        manifest,
        profile=args.h3_profile,
        count=args.requests,
    )
    network_payloads = [_network_payload(payload) for payload in payloads]
    offsets_s = _arrival_offsets(
        args.requests,
        process=args.arrival_process,
        request_rate=args.request_rate,
        seed=args.seed,
    )
    class_slos, class_slo_source = _load_class_slos(
        args.class_slo_file,
        request_classes=request_classes,
    )
    asyncio.run(_health(args.endpoint))

    if args.warmup_requests:
        warmup_count = min(args.warmup_requests, len(network_payloads))
        asyncio.run(
            _run_arrivals(
                endpoint=args.endpoint,
                payloads=network_payloads[:warmup_count],
                request_classes=request_classes[:warmup_count],
                offsets_s=[0.0] * warmup_count,
                max_tokens=args.max_tokens,
                key_prefix="warmup",
            )
        )

    memory_sampler = _DeviceProcessMemorySampler()
    memory_sampler.start()
    try:
        run = asyncio.run(
            _run_arrivals(
                endpoint=args.endpoint,
                payloads=network_payloads,
                request_classes=request_classes,
                offsets_s=offsets_s,
                max_tokens=args.max_tokens,
                key_prefix="formal",
            )
        )
    finally:
        process_device_memory = memory_sampler.stop()

    git = collect_git_metadata(REPO_ROOT, strict=True)
    server_config_path = None if args.server_config is None else Path(args.server_config)
    record = {
        "schema_version": 1,
        "record_type": "prism_network_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git.commit,
        "git_dirty": git.dirty,
        "hardware": collect_gpu_metadata().environment_dict(),
        "transport": {
            "protocol": "HTTP/1.1 + SSE",
            "endpoint": args.endpoint,
            "client": f"aiohttp/{aiohttp.__version__}",
            "server_config": (
                None
                if server_config_path is None
                else {
                    "path": str(server_config_path.resolve()),
                    "sha256": hashlib.sha256(server_config_path.read_bytes()).hexdigest(),
                }
            ),
        },
        "workload": {
            "manifest": str(Path(args.manifest).resolve()),
            "h3_contract": h3_contract,
            "requests": args.requests,
            "warmup_requests": args.warmup_requests,
            "max_tokens": args.max_tokens,
            "request_classes": request_classes,
            "network_payload_sha256": _canonical_sha256(network_payloads),
        },
        "arrival": {
            "process": args.arrival_process,
            "request_rate_per_s": args.request_rate,
            "seed": args.seed,
            "offsets_s": offsets_s,
            "trace_sha256": _canonical_sha256(
                {
                    "classes": request_classes,
                    "offsets_s": offsets_s,
                }
            ),
        },
        "protocol_conformance": _h3_conformance(h3_contract, args),
        "class_slo_source": class_slo_source,
        "run": run,
        "output_token_ids_sha256": _canonical_sha256(
            [record["token_ids"] for record in run["requests"]]
        ),
        "summary": _summary(run, class_slos),
        "memory": {"device_process": process_device_memory},
    }
    if record["summary"]["counts"]["completed"] != args.requests:
        raise RuntimeError("network H3 run did not complete every scheduled request")
    output = json.dumps(record, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"wrote network record to {output_path}", file=sys.stderr)
    print(output)


if __name__ == "__main__":
    main()
