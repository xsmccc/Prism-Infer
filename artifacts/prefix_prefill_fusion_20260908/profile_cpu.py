"""Attribute CPU time for one isolated Prefix-hit Prefill step (not a benchmark)."""

import argparse
import cProfile
import json
import pstats
from pathlib import Path
from time import perf_counter

from PIL import Image

from benchmarks.bench_serving_preprocessing import COLORS, LONG_PROMPT, PRIME_PROMPT
from prism_infer import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = LLM(
        "/home/lcpu/87120912/models/Qwen3-VL-8B-Instruct", **json.loads(args.config.read_text())
    )
    images = [Image.new("RGB", (448, 448), color) for color in COLORS]
    records = []
    prompts = (
        [PRIME_PROMPT]
        + [f"Describe the main color of image {i}. Answer with one color word." for i in (2, 3, 4)]
        + [LONG_PROMPT]
    )
    try:
        for index, prompt in enumerate(prompts):
            engine.add_images_request(
                prompt, images, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
            )
            profiler = cProfile.Profile()
            start = perf_counter()
            profiler.enable()
            step = engine.step_result()
            profiler.disable()
            elapsed = (perf_counter() - start) * 1000
            stats = pstats.Stats(profiler).stats
            records.append(
                {
                    "index": index,
                    "scheduled_tokens": step.plan.num_scheduled_tokens,
                    "step_ms": elapsed,
                    "functions": [
                        {
                            "file": k[0],
                            "line": k[1],
                            "name": k[2],
                            "primitive_calls": v[0],
                            "calls": v[1],
                            "self_ms": v[2] * 1000,
                            "cumulative_ms": v[3] * 1000,
                        }
                        for k, v in stats.items()
                    ],
                }
            )
            while not engine.is_finished():
                engine.step_result()
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        for r in records[1:]:
            print(r["index"], r["scheduled_tokens"], r["step_ms"])
            for f in sorted(r["functions"], key=lambda f: f["self_ms"], reverse=True)[:10]:
                print(round(f["self_ms"], 3), f["calls"], f["name"], f["file"])
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
