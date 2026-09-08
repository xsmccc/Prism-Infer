"""Isolate fusion cost after fixing the leaked default-device mode."""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from PIL import Image

import prism_infer.layers.attention as attention_module
import prism_infer.ops.qk_rmsnorm as qk_module
from benchmarks.bench_serving_preprocessing import COLORS, PRIME_PROMPT
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
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    rows = []
    try:
        engine.add_images_request(PRIME_PROMPT, images, sampling)
        while not engine.is_finished():
            engine.step_result()
        # The benchmark alone switches dispatch in a single, otherwise idle
        # engine. No new runtime flag or scheduling change is introduced.
        for index, fused in enumerate(
            (False, True, False, True, True, False, False, True, True, False)
        ):
            attention_module.HAS_KV_GATHER_TRITON = fused
            qk_module.MIN_PREFILL_QK_RMSNORM_ROWS = 1 if fused else 1024
            engine.add_images_request(
                "Describe the main color of image 2. Answer with one color word.", images, sampling
            )
            start = perf_counter_ns()
            first = engine.step_result()
            elapsed_ms = (perf_counter_ns() - start) / 1e6
            tokens = list(first.execution.token_ids)
            while not engine.is_finished():
                tokens.extend(engine.step_result().execution.token_ids)
            rows.append(
                {
                    "fused": fused,
                    "warmup": index < 2,
                    "scheduled_tokens": first.plan.num_scheduled_tokens,
                    "prefill_step_ms": elapsed_ms,
                    "token_ids": tokens,
                }
            )
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows))
    finally:
        attention_module.HAS_KV_GATHER_TRITON = True
        qk_module.MIN_PREFILL_QK_RMSNORM_ROWS = 1
        engine.exit()


if __name__ == "__main__":
    main()
