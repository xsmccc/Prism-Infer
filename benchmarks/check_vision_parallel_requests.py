"""Measure output agreement on existing real multi-image MuirBench requests.

This is a small paired implementation check, not the official MuirBench score.
Run once per vision mode against the same materialized records and model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from benchmarks.bench_vision_parallel import _run_requests
from prism_infer import LLM, SamplingParams
from prism_infer.analysis.quality_metrics import parse_muirbench_response
from prism_infer.analysis.working_set_quality import build_muirbench_media_first_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--vision-mode", choices=("replicated", "data"), required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records_path = args.materialized_root / "records/muirbench_test.final.jsonl"
    selected = []
    with records_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if 2 <= len(record["media"]) <= 8:
                selected.append(record)
            if len(selected) == args.limit:
                break
    options = {
        "tensor_parallel_size": 2,
        "vision_encoder_parallel_mode": args.vision_mode,
        "execution_backend": "cuda_graph",
        "max_model_len": 4096,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 4,
        "num_kvcache_blocks": 64,
        "kvcache_block_size": 256,
        "compression_mode": "scaled_fp8_kv",
        "enable_prefix_caching": False,
        "enable_visual_embedding_cache": False,
        "enable_chunked_prefill": False,
        "image_max_pixels": 448 * 448,
        "paged_decode_block_n": 256,
        "enable_fused_qk_rmsnorm": True,
        "enable_fused_qk_mrope": True,
        "enable_fused_add_rmsnorm": True,
        "enable_packed_kv_projection": True,
    }
    report = {
        "model": args.model,
        "options": options,
        "selection": "first records with 2 through 8 images, in existing materialization order",
        "records_file": str(records_path),
        "sampling": {"temperature": 0, "max_tokens": 16, "ignore_eos": True},
        "requests": [],
    }
    llm = LLM(args.model, **options)
    report["eos_token_id"] = llm.tokenizer.eos_token_id
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    try:
        for record in selected:
            images = []
            try:
                for media in record["media"]:
                    path = args.materialized_root / media["materialized_path"]
                    with Image.open(path) as source:
                        images.append(source.convert("RGB").copy())
                prompt = build_muirbench_media_first_prompt(
                    record["question"], record["options"], expected_media_count=len(images)
                )
                row = _run_requests(
                    llm,
                    images,
                    [prompt],
                    sampling,
                    phase="muirbench_cold",
                    prefix_enabled=False,
                    output_rows=report["requests"],
                    replica_id="0",
                    interleaved=True,
                )[0]
                answer_ids = row["token_ids"]
                if llm.tokenizer.eos_token_id in answer_ids:
                    answer_ids = answer_ids[: answer_ids.index(llm.tokenizer.eos_token_id)]
                text = llm.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                parsed = parse_muirbench_response(text, record["options"])
                row.update(
                    {
                        "sample_id": record["sample_id"],
                        "answer": record["answer"],
                        "prediction": text,
                        "answer_token_ids": answer_ids,
                        "parsed_answer": parsed.label,
                        "strict_correct": parsed.label == record["answer"],
                        "media_paths": [media["materialized_path"] for media in record["media"]],
                    }
                )
                print(json.dumps({"sample_id": row["sample_id"], "prediction": text}), flush=True)
            finally:
                for image in images:
                    image.close()
        report["status"] = "complete"
    finally:
        llm.exit()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
