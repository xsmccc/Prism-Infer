"""Actual Qwen3-VL requests for the repaired admission/cache/pressure paths."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from prism_infer import LLM, SamplingParams


def drain(engine, records):
    while not engine.is_finished():
        step = engine.step_result()
        records.append(
            {
                "phase": step.plan.phase.value,
                "requests": list(step.plan.sequence_ids),
                "scheduled_tokens": step.plan.num_scheduled_tokens,
                "tokens": list(step.execution.token_ids),
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = dict(
        execution_backend="cuda_graph",
        compression_mode="scaled_fp8_kv",
        max_num_seqs=2,
        kvcache_block_size=256,
        cpu_kv_cache_ratio=0.0,
        enable_fused_qk_rmsnorm=True,
        enable_fused_qk_mrope=True,
        enable_fused_add_rmsnorm=True,
        enable_packed_kv_projection=True,
        paged_decode_block_n=256,
    )
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cases": {}}
    engine = LLM(
        args.model,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        num_kvcache_blocks=4,
        enable_chunked_prefill=True,
        max_chunk_size=256,
        enable_prefix_caching=False,
        **common,
    )
    try:
        token = engine.tokenizer.encode("hello", add_special_tokens=False)[0]
        first = engine.add_request(
            [token] * 254, SamplingParams(max_tokens=8, ignore_eos=True, temperature=0)
        )
        engine.step_result()
        second = engine.add_request(
            [token] * 768, SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
        )
        steps = []
        drain(engine, steps)
        a, b = engine.scheduler.get_request(first), engine.scheduler.get_request(second)
        assert a.is_finished and b.is_finished
        assert len(a.completion_token_ids) == 8 and len(b.completion_token_ids) == 1
        assert engine.scheduler.recompute_preemptions == 0
        report["cases"]["chunked_pressure"] = {
            "steps": steps,
            "first": a.completion_token_ids,
            "second": b.completion_token_ids,
        }
    finally:
        engine.exit()

    images = [
        Image.new("RGB", (448, 448), color)
        for color in ((220, 40, 40), (40, 190, 60), (50, 80, 230))
    ]
    options = dict(
        max_model_len=4096,
        max_num_batched_tokens=4096,
        num_kvcache_blocks=64,
        image_max_pixels=448 * 448,
        **common,
    )
    sampling = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)
    prompt = "Name the main color of the first image. Answer with one color word."
    engine = LLM(args.model, enable_visual_embedding_cache=True, **options)
    try:
        assert not engine.config.enable_chunked_prefill
        outputs = []
        cache_states = []
        for question, group in (
            (prompt, images),
            ("Name the main color of the second image.", images),
            (prompt, [images[1], images[0], images[2]]),
        ):
            request_id = engine.add_images_request(question, group, sampling)
            drain(engine, [])
            outputs.append(engine.scheduler.get_request(request_id).completion_token_ids)
            cache_states.append(engine.visual_embedding_cache_metadata())
        assert cache_states[1]["host_cache"]["hits"] == cache_states[0]["host_cache"]["hits"]
        assert cache_states[2]["host_cache"]["hits"] > cache_states[1]["host_cache"]["hits"]
        before_rejection = engine.visual_embedding_cache_metadata()
        try:
            engine.add_images_request(
                "hello " * 5000, [Image.new("RGB", (448, 448), (150, 80, 30))], sampling
            )
        except ValueError as error:
            assert "exceeds max_model_len" in str(error)
        else:
            raise AssertionError("oversized request was not rejected")
        assert engine.visual_embedding_cache_metadata() == before_rejection
        report["cases"]["visual_cache"] = {
            "outputs": outputs,
            "cache_states": cache_states,
            "rejected_before_vision": True,
        }
    finally:
        engine.exit()
    engine = LLM(
        args.model, enable_visual_embedding_cache=False, enable_prefix_caching=False, **options
    )
    try:
        reference_id = engine.add_images_request(
            prompt, [images[1], images[0], images[2]], sampling
        )
        drain(engine, [])
        reference = engine.scheduler.get_request(reference_id).completion_token_ids
        assert reference == outputs[2]
        report["cases"]["visual_cache"]["reordered_matches_uncached"] = True
    finally:
        engine.exit()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": list(report["cases"]), "status": "passed"}))


if __name__ == "__main__":
    main()
