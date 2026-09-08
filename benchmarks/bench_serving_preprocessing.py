"""Measure real HTTP prefix hits and decode interrupted by a cold image request.

Run the same client against the original and updated server in separate runs.
SSE tokens completed in one network read share its observation timestamp.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns

import aiohttp
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.vision_parallel_report import save_report

COLORS = (
    (220, 50, 50),
    (50, 180, 80),
    (60, 100, 220),
    (220, 180, 40),
    (170, 70, 200),
    (40, 190, 190),
    (230, 120, 40),
    (120, 120, 120),
)
PRIME_PROMPT = "Describe the main color of the first image. Answer with one color word."
LONG_PROMPT = "Describe each image and compare all their colors in detail."


def image_group(group: int) -> list[str]:
    """Create a reproducible, content-distinct eight-image fixture, outside timing."""
    encoded = []
    for color in COLORS:
        rgb = tuple((channel + 17 * group) % 256 for channel in color)
        with Image.new("RGB", (448, 448), rgb) as image, io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            encoded.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return encoded


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


async def request(
    session: aiohttp.ClientSession,
    url: str,
    media: list[str],
    row: dict,
    *,
    trigger: asyncio.Event | None = None,
) -> None:
    payload = {
        "request_id": row["request_id"],
        "modality": "image",
        "media": media,
        "prompt": row["prompt"],
        "stream": True,
        "sampling": {"temperature": 0, "max_tokens": row["max_tokens"], "ignore_eos": True},
    }
    row.update({"token_ids": [], "token_arrival_ns": [], "network_chunks": [], "status": "running"})
    row["client_start_ns"] = perf_counter_ns()
    pending = b""
    async with session.post(url + "/v1/generate", json=payload) as response:
        response.raise_for_status()
        async for chunk in response.content.iter_any():
            observed_ns = perf_counter_ns()
            pending += chunk
            events = []
            while b"\n\n" in pending:
                frame, pending = pending.split(b"\n\n", 1)
                data = b"\n".join(
                    line[6:] for line in frame.splitlines() if line.startswith(b"data: ")
                )
                if not data:
                    continue
                event = json.loads(data)
                events.append(event)
                if event["event"] == "error":
                    raise RuntimeError(event["error"])
                if event["event"] == "token":
                    row["token_ids"].append(event["token_id"])
                    row["token_arrival_ns"].append(observed_ns)
                    if trigger is not None and len(row["token_ids"]) >= 3:
                        trigger.set()
                elif event["event"] == "done":
                    row["client_finish_ns"] = observed_ns
                    row["final_token_ids"] = event["token_ids"]
                    row["finish_reason"] = event["finish_reason"]
            row["network_chunks"].append(
                {"arrival_ns": observed_ns, "bytes": len(chunk), "events": events}
            )
    if "client_finish_ns" not in row or len(row["token_ids"]) < 2:
        raise RuntimeError("SSE ended without a completed multi-token generation")
    arrivals = row["token_arrival_ns"]
    row["itl_ms"] = [(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:], strict=False)]
    row["ttft_ms"] = (arrivals[0] - row["client_start_ns"]) / 1e6
    row["e2e_ms"] = (row["client_finish_ns"] - row["client_start_ns"]) / 1e6
    row["tpot_ms"] = statistics.mean(row["itl_ms"])
    row["max_itl_ms"] = max(row["itl_ms"])
    row["p95_itl_ms"] = percentile(row["itl_ms"], 0.95)
    row["stream_matches_final_ids"] = row["token_ids"] == row["final_token_ids"]
    row["status"] = "finished"


async def run(args, report: dict) -> None:
    media = [image_group(group) for group in range(args.repeat + 1)]
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(args.base_url + "/health") as response:
            response.raise_for_status()

        def row(phase, prompt, max_tokens=16, media_group=0):
            result = {
                "phase": phase,
                "request_id": f"{args.label}-{len(report['requests'])}",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "media_group": media_group,
            }
            report["requests"].append(result)
            return result

        async def single(phase, prompt, max_tokens=16):
            current = row(phase, prompt, max_tokens)
            await request(session, args.base_url, media[0], current)
            save_report(args.output, report)

        await single("warmup", PRIME_PROMPT)
        for index in range(args.repeat):
            await single(
                "hot_changed_question",
                f"Describe the main color of image {index + 2}. Answer with one color word.",
            )
        await single("decode_alone", LONG_PROMPT, args.long_tokens)
        for index in range(args.repeat):
            trigger = asyncio.Event()
            long_row = row("decode_with_cold_image", LONG_PROMPT, args.long_tokens)
            decoding = asyncio.create_task(
                request(session, args.base_url, media[0], long_row, trigger=trigger)
            )
            ready = asyncio.create_task(trigger.wait())
            await asyncio.wait((decoding, ready), return_when=asyncio.FIRST_COMPLETED)
            if not trigger.is_set():
                ready.cancel()
                await asyncio.gather(ready, return_exceptions=True)
                await decoding
                raise RuntimeError(
                    "long generation finished before the cold-request injection point"
                )
            cold_row = row("injected_cold_image", PRIME_PROMPT, media_group=index + 1)
            cold_row["long_request_id"] = long_row["request_id"]
            cold_row["long_tokens_observed_at_submit"] = len(long_row["token_ids"])
            cold_row["long_already_finished_at_submit"] = "client_finish_ns" in long_row
            if cold_row["long_already_finished_at_submit"]:
                await asyncio.gather(decoding, ready)
                raise RuntimeError("SSE was coalesced past the live-decode injection point")
            cold = asyncio.create_task(request(session, args.base_url, media[index + 1], cold_row))
            try:
                await asyncio.gather(decoding, cold, ready)
            finally:
                save_report(args.output, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8137")
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--long-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.long_tokens <= 3:
        parser.error("repeat must be positive and long-tokens must exceed the injection point (3)")
    args.base_url = args.base_url.rstrip("/")
    report = {
        "label": args.label,
        "base_url": args.base_url,
        "requests": [],
        "status": "running",
        "protocol": {
            "fixture": "eight 448x448 solid RGB images; colors shifted by 17 per cold group",
            "colors": COLORS,
            "repeat": args.repeat,
            "long_tokens": args.long_tokens,
            "timing": (
                "HTTP client; JSON upload, server image decode, preprocessing, queueing included"
            ),
            "token_time": "observed network read completing each SSE event; no interpolation",
            "cold_injection": "after at least 3 observed tokens of a live long decode",
        },
    }
    save_report(args.output, report)
    try:
        asyncio.run(run(args, report))
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["phase_summaries"] = {
            phase: {
                key: statistics.median(row[key] for row in rows)
                for key in ("ttft_ms", "e2e_ms", "tpot_ms", "max_itl_ms", "p95_itl_ms")
            }
            for phase in dict.fromkeys(row["phase"] for row in report["requests"])
            if (
                rows := [
                    row
                    for row in report["requests"]
                    if row["phase"] == phase and row.get("status") == "finished"
                ]
            )
        }
        save_report(args.output, report)


if __name__ == "__main__":
    main()
