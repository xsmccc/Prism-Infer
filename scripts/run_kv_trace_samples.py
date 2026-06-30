#!/usr/bin/env python3
"""运行 P4 KV trace 三类样例并验证 trace on/off greedy 一致。

样例:
- single_image_description
- single_image_detail_qa
- multi_image_comparison

该脚本需要本地 Qwen3-VL 模型和 CUDA。它不会把未运行样例写成 PASS；
任何 token 不一致都会抛出 AssertionError。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_infer import LLM
from prism_infer.analysis.kv_trace import (
    format_summary_markdown,
    kv_trace,
    read_trace_jsonl,
    render_summary_svg,
    summarize_trace,
)
from prism_infer.sampling_params import SamplingParams


def _image(color: tuple[int, int, int], *, label: str) -> Image.Image:
    image = Image.new("RGB", (448, 448), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 408, 408], outline=(255, 255, 255), width=8)
    draw.text((64, 64), label, fill=(255, 255, 255))
    return image


def _sample_cases() -> list[dict]:
    blue = _image((70, 120, 210), label="BLUE")
    red = _image((210, 90, 70), label="RED")
    green = _image((70, 170, 110), label="GREEN")
    return [
        {
            "name": "single_image_description",
            "kind": "image",
            "prompt": "Describe this image in one short sentence.",
            "image": blue,
        },
        {
            "name": "single_image_detail_qa",
            "kind": "image",
            "prompt": "What color dominates the image? Answer briefly.",
            "image": red,
        },
        {
            "name": "multi_image_comparison",
            "kind": "images",
            "prompt": "Compare the dominant colors in these images.",
            "images": [blue, green],
        },
    ]


def _new_llm(model_path: str, max_model_len: int, max_num_batched_tokens: int) -> LLM:
    return LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=False,
    )


def _run_case(llm: LLM, case: dict, sampling: SamplingParams) -> dict:
    if case["kind"] == "image":
        return llm.generate_vl(
            case["prompt"],
            case["image"],
            sampling,
            use_tqdm=False,
        )
    if case["kind"] == "images":
        return llm.generate_images(
            case["prompt"],
            case["images"],
            sampling,
            use_tqdm=False,
        )
    raise ValueError(f"unsupported case kind: {case['kind']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prism-Infer P4 KV trace sample cases")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "PRISM_MODEL_PATH",
            "/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/kv_trace_samples"))
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--top-k-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P4 KV trace samples require CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    results = []

    for case in _sample_cases():
        trace_path = args.output_dir / f"{case['name']}.jsonl"
        summary_json_path = args.output_dir / f"{case['name']}.summary.json"
        summary_md_path = args.output_dir / f"{case['name']}.summary.md"
        summary_svg_path = args.output_dir / f"{case['name']}.summary.svg"

        llm_off = _new_llm(args.model, args.max_model_len, args.max_num_batched_tokens)
        try:
            output_off = _run_case(llm_off, case, sampling)
        finally:
            llm_off.exit()

        llm_on = _new_llm(args.model, args.max_model_len, args.max_num_batched_tokens)
        try:
            with kv_trace(
                trace_path,
                metadata={"case": case["name"], "kind": case["kind"]},
                top_k_tokens=args.top_k_tokens,
            ):
                output_on = _run_case(llm_on, case, sampling)
        finally:
            llm_on.exit()

        if output_on["token_ids"] != output_off["token_ids"]:
            raise AssertionError(
                f"trace on/off token mismatch for {case['name']}: "
                f"off={output_off['token_ids']}, on={output_on['token_ids']}"
            )

        records = read_trace_jsonl(trace_path)
        summary = summarize_trace(records)
        summary_json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary_md_path.write_text(format_summary_markdown(summary), encoding="utf-8")
        summary_svg_path.write_text(render_summary_svg(summary), encoding="utf-8")
        result = {
            "case": case["name"],
            "kind": case["kind"],
            "token_ids": output_on["token_ids"],
            "trace_path": str(trace_path),
            "summary_json": str(summary_json_path),
            "summary_markdown": str(summary_md_path),
            "summary_svg": str(summary_svg_path),
            "num_layer_records": summary["num_layer_records"],
            "num_steps": summary["num_steps"],
            "layers": len(summary["layers"]),
            "phases": summary["phases"],
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    manifest = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "case_count": len(results),
        "results": results,
        "result": "PASS",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
