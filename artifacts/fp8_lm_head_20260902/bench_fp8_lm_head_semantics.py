"""Measure Prism's FP8 LM-head candidate path against BF16 and FP32 references."""

from __future__ import annotations

import argparse
import json
from inspect import signature
from pathlib import Path
from statistics import mean, median

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from prism_infer.engine.vl_inputs import (
    prepare_image_inputs,
    prepare_interleaved_image_inputs,
    prepare_video_inputs,
)
from prism_infer.models.qwen3_vl import compile_decode_fp8_lm_head
from prism_infer.ops.selective_topk import (
    rerank_greedy_candidates,
    selective_topk_indices,
)


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
TOP_K = 64


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (448, 448), color=color)


def _cases() -> list[dict[str, object]]:
    red = _image((220, 50, 50))
    green = _image((50, 180, 80))
    blue = _image((60, 100, 220))
    video = [
        _image((32 + step * 24, 96 + step * 8, 160 - step * 16))
        for step in range(4)
    ]
    eight = [
        red,
        green,
        blue,
        _image((220, 180, 40)),
        _image((170, 70, 200)),
        _image((40, 190, 190)),
        _image((230, 120, 40)),
        _image((120, 120, 120)),
    ]
    return [
        {
            "kind": "text",
            "payload": None,
            "prompts": [
                "Explain why decode at batch size one is usually memory bound.",
                "Give three practical uses of CUDA Graph in model inference.",
                "Compare prefix caching with an encoder-output cache.",
                "用中文解释分页 KV Cache 的基本工作流程。",
                "A train travels 120 km in 90 minutes. Compute its average speed.",
                "Write a concise explanation of grouped-query attention.",
            ],
        },
        {
            "kind": "image",
            "payload": red,
            "prompts": [
                "Describe the dominant color and visual structure.",
                "What is the most salient property of this image?",
                "Is the image visually complex? Explain briefly.",
                "Give a one-sentence caption for this image.",
                "What object or pattern can be inferred from the image?",
                "Describe this image in Chinese.",
            ],
        },
        {
            "kind": "image",
            "payload": [red, green],
            "prompts": [
                "Compare the two images and state their main difference.",
                "Describe the first image and then the second image.",
                "Which image appears greener and why?",
                "Summarize the shared structure of these images.",
                "Give a short ordered caption for both images.",
                "用中文比较这两张图片。",
            ],
        },
        {
            "kind": "video",
            "payload": video,
            "prompts": [
                "Describe how the colors change over time.",
                "What is the initial state and final state of this sequence?",
                "Summarize the temporal trend in one sentence.",
                "Does the sequence become brighter or darker?",
                "Describe the most important transition between frames.",
                "用中文描述这段视频的变化。",
            ],
        },
        {
            "kind": "image",
            "payload": eight,
            "prompts": [
                "Compare all eight images carefully. Describe the important details in each image, then identify similarities, differences, and any cross-image pattern.",
                "Summarize the ordered color pattern across all eight images.",
                "Group the eight images by visual similarity and explain the grouping.",
            ],
        },
    ]


def _muir_cases(
    plan_path: Path,
    materialized_root: Path,
    max_prompts: int,
) -> list[dict[str, object]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cases = []
    groups = sorted(
        plan["groups"],
        key=lambda group: (len(group["samples"]) == 1, group["group_id"]),
    )
    for group in groups:
        paths = [materialized_root / media["materialized_path"] for media in group["media"]]
        for sample in group["samples"]:
            cases.append(
                {
                    "kind": "muirbench",
                    "payload": (paths, plan["prompt_layout"]["image_marker"]),
                    "prompts": [sample["source_prompt"]],
                }
            )
            if len(cases) >= max_prompts:
                return cases
    return cases


def _mm_token_type_ids(model, input_ids: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(input_ids, dtype=torch.long)
    image_token_id = getattr(model.config, "image_token_id", None)
    video_token_id = getattr(model.config, "video_token_id", None)
    if image_token_id is not None:
        result[input_ids == int(image_token_id)] = 1
    if video_token_id is not None:
        result[input_ids == int(video_token_id)] = 2
    return result


def _prepare_inputs(processor, model, kind: str, prompt: str, payload) -> dict:
    if kind == "text":
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = processor(text=[text], return_tensors="pt")
        result = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    elif kind == "image":
        inputs = prepare_image_inputs(processor, prompt, payload)
        result = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
        }
    elif kind == "video":
        inputs = prepare_video_inputs(processor, prompt, payload)
        result = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values_videos": inputs.pixel_values_videos,
            "video_grid_thw": inputs.video_grid_thw,
        }
    elif kind == "muirbench":
        paths, image_marker = payload
        images = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB").copy())
            inputs = prepare_interleaved_image_inputs(
                processor,
                prompt,
                images,
                image_marker=image_marker,
            )
        finally:
            for image in images:
                image.close()
        result = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
        }
    else:
        raise ValueError(kind)
    if "mm_token_type_ids" in signature(model.forward).parameters:
        result["mm_token_type_ids"] = _mm_token_type_ids(model, result["input_ids"])
    return {key: value.cuda(non_blocking=True) for key, value in result.items()}


def _prepare_fp8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_size, hidden_size = weight.shape
    weight_fp8 = torch.empty(
        vocab_size,
        hidden_size,
        device=weight.device,
        dtype=torch.float8_e4m3fn,
    )
    scale_t = torch.empty(1, vocab_size, device=weight.device, dtype=torch.float32)
    for start in range(0, vocab_size, 4096):
        end = min(start + 4096, vocab_size)
        chunk = weight[start:end].float()
        scale = (chunk.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
        weight_fp8[start:end] = (
            (chunk / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        )
        scale_t[:, start:end] = scale.t()
    return weight_fp8.t(), scale_t


def _fp32_projection(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    parts = []
    hidden_fp32 = hidden.float()
    for start in range(0, weight.shape[0], 4096):
        parts.append(F.linear(hidden_fp32, weight[start : start + 4096].float()))
    return torch.cat(parts, dim=-1)


def _overlap_at_five(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_ids = set(reference.topk(5).indices.tolist())
    candidate_ids = set(candidate.topk(5).indices.tolist())
    return len(reference_ids & candidate_ids) / 5.0


@torch.inference_mode()
def _measure_prompt(
    *,
    model,
    processor,
    compiled_fp8,
    weight: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    kind: str,
    prompt: str,
    payload,
    max_new_tokens: int,
) -> dict[str, object]:
    inputs = _prepare_inputs(processor, model, kind, prompt, payload)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    captures = []

    def capture_lm_head(_module, module_inputs, module_output) -> None:
        hidden = module_inputs[0][:, -1, :].detach()
        fp32_logits = _fp32_projection(hidden, weight)
        captures.append(
            (
                hidden.clone(),
                module_output[:, -1, :].detach().clone(),
                fp32_logits.detach().clone(),
            )
        )
        return fp32_logits.unsqueeze(1)

    hook = model.lm_head.register_forward_hook(capture_lm_head)
    try:
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            use_cache=True,
        )
    finally:
        hook.remove()
    generated_ids = generated.sequences[0, prompt_tokens:].tolist()
    if len(captures) != len(generated_ids):
        raise RuntimeError(
            f"LM-head calls and generated tokens differ: {len(captures)} != "
            f"{len(generated_ids)}"
        )
    rows = []
    first_bf16_divergence = None
    first_fp32_divergence = None
    for step, (hidden, native_logits, reference_logits) in enumerate(captures):
        hidden = hidden.contiguous()
        bf16_logits = native_logits.squeeze(0)
        fp32_logits = reference_logits.squeeze(0)
        approximate_logits = compiled_fp8(hidden, weight_fp8_t, weight_scale).squeeze(0)
        top_ids = selective_topk_indices(approximate_logits.unsqueeze(0), k=TOP_K)
        candidate_id = int(
            rerank_greedy_candidates(top_ids, hidden, weight).item()
        )
        approximate_id = int(approximate_logits.argmax().item())
        bf16_id = int(bf16_logits.argmax().item())
        fp32_id = int(fp32_logits.argmax().item())
        top_id_set = set(top_ids[0].tolist())
        selected_scores = approximate_logits[top_ids[0]]
        fp32_selected_rank = None
        if fp32_id in top_id_set:
            fp32_score = approximate_logits[fp32_id]
            fp32_selected_rank = int((selected_scores > fp32_score).sum().item()) + 1
        bf16_match = candidate_id == bf16_id
        fp32_match = candidate_id == fp32_id
        if not bf16_match and first_bf16_divergence is None:
            first_bf16_divergence = step
        if not fp32_match and first_fp32_divergence is None:
            first_fp32_divergence = step
        fp32_top2 = fp32_logits.topk(2).values
        approximate_f = approximate_logits.float()
        bf16_f = bf16_logits.float()
        rows.append(
            {
                "step": step,
                "generated_id": int(generated_ids[step]),
                "bf16_id": bf16_id,
                "fp32_id": fp32_id,
                "candidate_id": candidate_id,
                "approximate_id": approximate_id,
                "bf16_winner_in_top64": bf16_id in top_id_set,
                "fp32_winner_in_top64": fp32_id in top_id_set,
                "fp32_approximate_rank_in_top64": fp32_selected_rank,
                "approximate_top1_matches_fp32": approximate_id == fp32_id,
                "rerank_changed_top1": candidate_id != approximate_id,
                "candidate_matches_bf16": bf16_match,
                "candidate_matches_fp32": fp32_match,
                "bf16_matches_fp32": bf16_id == fp32_id,
                "generated_matches_bf16": int(generated_ids[step]) == bf16_id,
                "fp32_margin": float((fp32_top2[0] - fp32_top2[1]).item()),
                "cosine_similarity": float(
                    F.cosine_similarity(approximate_f, bf16_f, dim=0).item()
                ),
                "kl_bf16_to_fp8": float(
                    F.kl_div(
                        F.log_softmax(approximate_f, dim=0),
                        F.log_softmax(bf16_f, dim=0),
                        reduction="sum",
                        log_target=True,
                    ).item()
                ),
                "mean_abs_logit_diff": float((approximate_f - bf16_f).abs().mean().item()),
                "max_abs_logit_diff": float((approximate_f - bf16_f).abs().max().item()),
                "top5_overlap": _overlap_at_five(bf16_f, approximate_f),
            }
        )
    del generated, inputs
    torch.cuda.empty_cache()
    return {
        "kind": kind,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(rows),
        "generated_text": processor.decode(generated_ids, skip_special_tokens=False),
        "first_bf16_divergence": first_bf16_divergence,
        "first_fp32_divergence": first_fp32_divergence,
        "rows": rows,
    }


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    rows = [row for record in records for row in record["rows"]]
    failures = [row for row in rows if not row["candidate_matches_fp32"]]
    fp32_ranks = [
        row["fp32_approximate_rank_in_top64"]
        for row in rows
        if row["fp32_approximate_rank_in_top64"] is not None
    ]
    return {
        "prompts": len(records),
        "decode_rows": len(rows),
        "request_exact_vs_bf16": mean(
            record["first_bf16_divergence"] is None for record in records
        ),
        "request_exact_vs_fp32": mean(
            record["first_fp32_divergence"] is None for record in records
        ),
        "top64_recall_bf16": mean(row["bf16_winner_in_top64"] for row in rows),
        "top64_recall_fp32": mean(row["fp32_winner_in_top64"] for row in rows),
        "reranked_top1_agreement_bf16": mean(
            row["candidate_matches_bf16"] for row in rows
        ),
        "reranked_top1_agreement_fp32": mean(
            row["candidate_matches_fp32"] for row in rows
        ),
        "approximate_top1_agreement_fp32": mean(
            row["approximate_top1_matches_fp32"] for row in rows
        ),
        "rerank_changed_top1_rate": mean(row["rerank_changed_top1"] for row in rows),
        "mean_fp32_winner_rank_in_fp8_top64": mean(fp32_ranks) if fp32_ranks else None,
        "worst_fp32_winner_rank_in_fp8_top64": max(fp32_ranks) if fp32_ranks else None,
        "bf16_top1_agreement_fp32": mean(row["bf16_matches_fp32"] for row in rows),
        "generated_top1_agreement_bf16": mean(
            row["generated_matches_bf16"] for row in rows
        ),
        "mean_cosine_similarity": mean(row["cosine_similarity"] for row in rows),
        "mean_kl_bf16_to_fp8": mean(row["kl_bf16_to_fp8"] for row in rows),
        "mean_abs_logit_diff": mean(row["mean_abs_logit_diff"] for row in rows),
        "mean_of_max_abs_logit_diff": mean(row["max_abs_logit_diff"] for row in rows),
        "worst_max_abs_logit_diff": max(row["max_abs_logit_diff"] for row in rows),
        "mean_top5_overlap": mean(row["top5_overlap"] for row in rows),
        "fp32_failure_count": len(failures),
        "fp32_failure_margins": [row["fp32_margin"] for row in failures],
    }


@torch.inference_mode()
def _benchmark_graph_paths(
    weight: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    compiled_fp8,
    iterations: int = 1000,
) -> dict[str, object]:
    hidden = torch.randn(1, weight.shape[1], device="cuda", dtype=torch.bfloat16)

    for _ in range(5):
        F.linear(hidden, weight).argmax(dim=-1)
        approximate = compiled_fp8(hidden, weight_fp8_t, weight_scale)
        candidates = selective_topk_indices(approximate, k=TOP_K)
        rerank_greedy_candidates(candidates, hidden, weight)
    torch.cuda.synchronize()

    baseline_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(baseline_graph):
        baseline_token = F.linear(hidden, weight).argmax(dim=-1)

    candidate_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(candidate_graph):
        approximate = compiled_fp8(hidden, weight_fp8_t, weight_scale)
        candidates = selective_topk_indices(approximate, k=TOP_K)
        candidate_token = rerank_greedy_candidates(candidates, hidden, weight)

    def measure(graph: torch.cuda.CUDAGraph) -> float:
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iterations

    baseline_ms = [measure(baseline_graph) for _ in range(5)]
    candidate_ms = [measure(candidate_graph) for _ in range(5)]
    baseline_median = median(baseline_ms)
    candidate_median = median(candidate_ms)
    return {
        "shape": [1, int(weight.shape[1]), int(weight.shape[0])],
        "iterations_per_repeat": iterations,
        "repeats": 5,
        "baseline": "BF16 full-vocabulary projection plus argmax",
        "candidate": "compiled FP8 projection plus top-64 plus FP32 rerank",
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "relative_reduction": (baseline_median - candidate_median) / baseline_median,
        "baseline_token": int(baseline_token.item()),
        "candidate_token": int(candidate_token.item()),
        "extra_fp8_weight_bytes": int(weight_fp8_t.numel() * weight_fp8_t.element_size()),
        "extra_scale_bytes": int(weight_scale.numel() * weight_scale.element_size()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--muir-plan", type=Path)
    parser.add_argument("--muir-root", type=Path)
    parser.add_argument("--muir-prompts", type=int, default=24)
    parser.add_argument("--muir-only", action="store_true")
    parser.add_argument("--speed-only", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).cuda().eval()
    model.requires_grad_(False)
    weight = model.lm_head.weight.detach()
    with torch.no_grad():
        weight_fp8_t, weight_scale = _prepare_fp8_weight(weight)
    compiled_fp8 = compile_decode_fp8_lm_head(
        mode="max-autotune-no-cudagraphs",
        emulate_precision_casts=True,
        force_same_precision=True,
    )
    compiled_fp8(
        torch.zeros(1, weight.shape[1], device="cuda", dtype=torch.bfloat16),
        weight_fp8_t,
        weight_scale,
    )
    torch.cuda.synchronize()

    if args.speed_only:
        result = {
            "schema_version": 1,
            "experiment": "fp8_lm_head_cuda_graph_microbenchmark",
            "model": args.model,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "result": _benchmark_graph_paths(
                weight,
                weight_fp8_t,
                weight_scale,
                compiled_fp8,
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result["result"], ensure_ascii=False, indent=2), flush=True)
        return

    cases = [] if args.muir_only else _cases()
    if args.muir_plan is not None:
        if args.muir_root is None:
            parser.error("--muir-root is required with --muir-plan")
        cases.extend(_muir_cases(args.muir_plan, args.muir_root, args.muir_prompts))

    records = []
    for case in cases:
        for prompt in case["prompts"]:
            if args.max_prompts is not None and len(records) >= args.max_prompts:
                break
            print(f"[{len(records) + 1}] {case['kind']}: {prompt}", flush=True)
            records.append(
                _measure_prompt(
                    model=model,
                    processor=processor,
                    compiled_fp8=compiled_fp8,
                    weight=weight,
                    weight_fp8_t=weight_fp8_t,
                    weight_scale=weight_scale,
                    kind=case["kind"],
                    prompt=prompt,
                    payload=case["payload"],
                    max_new_tokens=args.max_new_tokens,
                )
            )
        if args.max_prompts is not None and len(records) >= args.max_prompts:
            break

    result = {
        "schema_version": 1,
        "experiment": "fp8_lm_head_top64_fp32_rerank_semantics",
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "protocol": {
            "trajectory": "HF greedy with full-vocabulary FP32 LM-head accumulation",
            "max_new_tokens": args.max_new_tokens,
            "candidate_top_k": TOP_K,
            "reference_bf16": "native HF BF16 full-vocabulary LM-head logits",
            "reference_fp32": "BF16 hidden/weight with FP32 full-vocabulary accumulation",
        },
        "summary": _aggregate(records),
        "summary_by_kind": {
            kind: _aggregate([record for record in records if record["kind"] == kind])
            for kind in sorted({record["kind"] for record in records})
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
