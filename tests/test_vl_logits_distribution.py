"""P3.4 VL 生成轨迹 logits 分布与 perplexity 验证。"""

import gc

import pytest
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration

from conftest import get_model_path, require_transformers, with_hf_mm_token_type_ids
from prism_infer.engine.vl_inputs import prepare_image_inputs, prepare_video_inputs
from prism_infer.models.qwen3_vl import Qwen3VLForCausalLM
from prism_infer.models.qwen3_vl_position import get_qwen3_vl_rope_index_from_config
from test_processor_pipeline_video import demo_video_frames


pytestmark = [
    pytest.mark.model,
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.slow,
]

DTYPE = torch.bfloat16
DEVICE = "cuda"
MAX_TOKENS = 32
CHECKPOINTS = (8, 16, 32)
MAX_LOGIT_DIFF_TOL = 3e-1
MEAN_LOGIT_DIFF_TOL = 1e-2
PPL_DIFF_TOL = 1e-1


def _require_cuda() -> None:
    if torch.cuda.is_available():
        return
    pytest = __import__("pytest")
    pytest.skip("VL logits distribution test requires CUDA")


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (448, 448), color=color)


def _cases() -> list[dict]:
    return [
        {
            "name": "single-image",
            "prompt": "Describe this image.",
            "kind": "image",
            "payload": _image((100, 150, 200)),
        },
        {
            "name": "multi-image",
            "prompt": "Compare these images.",
            "kind": "image",
            "payload": [_image((100, 150, 200)), _image((200, 120, 80))],
        },
        {
            "name": "video",
            "prompt": "Describe this video.",
            "kind": "video",
            "payload": demo_video_frames(),
        },
    ]


def _load_our_from_hf_cpu(hf_cpu) -> Qwen3VLForCausalLM:
    """从 HF 模型权重构造 Prism-Infer 模型。"""

    default_device = torch.get_default_device()
    torch.set_default_device(DEVICE)
    try:
        our = Qwen3VLForCausalLM(config=hf_cpu.config, dtype=DTYPE).eval()
    finally:
        torch.set_default_device(default_device)

    our_sd = our.state_dict()
    hf_sd = hf_cpu.state_dict()
    loaded = 0
    missing = []
    for key in our_sd:
        if key in hf_sd:
            our_sd[key].copy_(hf_sd[key].to(DEVICE), non_blocking=True)
            loaded += 1
        else:
            missing.append(key)
    unexpected = [key for key in hf_sd if key not in our_sd]
    print(f"teacher-forced weight loaded: {loaded}/{len(our_sd)}")
    print(f"teacher-forced missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"teacher-forced unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    assert not missing
    return our


def _prepare_case(processor, config, case: dict):
    if case["kind"] == "image":
        inputs = prepare_image_inputs(processor, case["prompt"], case["payload"])
        position_ids, _ = get_qwen3_vl_rope_index_from_config(
            inputs.input_ids,
            config=config,
            image_grid_thw=inputs.image_grid_thw,
            attention_mask=inputs.attention_mask,
        )
        hf_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
        }
        our_kwargs = {
            "input_ids": inputs.input_ids,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
            "position_ids": position_ids,
        }
        token_count = inputs.image_token_count
    elif case["kind"] == "video":
        inputs = prepare_video_inputs(processor, case["prompt"], case["payload"])
        position_ids, _ = get_qwen3_vl_rope_index_from_config(
            inputs.input_ids,
            config=config,
            video_grid_thw=inputs.video_grid_thw,
            attention_mask=inputs.attention_mask,
        )
        hf_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values_videos": inputs.pixel_values_videos,
            "video_grid_thw": inputs.video_grid_thw,
        }
        our_kwargs = {
            "input_ids": inputs.input_ids,
            "pixel_values_videos": inputs.pixel_values_videos,
            "video_grid_thw": inputs.video_grid_thw,
            "position_ids": position_ids,
        }
        token_count = inputs.video_token_count
    else:
        raise ValueError(case["kind"])
    return inputs, hf_kwargs, our_kwargs, token_count


def _to_cuda_kwargs(kwargs: dict) -> dict:
    return {key: value.to(DEVICE) for key, value in kwargs.items()}


def _teacher_forced_hf_logits(hf_model, hf_kwargs: dict, generated_ids: list[int]) -> torch.Tensor:
    generated = torch.tensor([generated_ids], dtype=torch.long)
    full_input_ids = torch.cat([hf_kwargs["input_ids"], generated], dim=1)
    full_attention = torch.ones_like(full_input_ids)

    hf_full = dict(hf_kwargs)
    hf_full["input_ids"] = full_input_ids
    hf_full["attention_mask"] = full_attention
    hf_full = with_hf_mm_token_type_ids(hf_model, hf_full)

    with torch.inference_mode():
        return (
            hf_model(**_to_cuda_kwargs(hf_full)).logits[:, -MAX_TOKENS - 1 : -1, :].detach().cpu()
        )


def _teacher_forced_our_logits(
    our_model,
    config,
    our_kwargs: dict,
    generated_ids: list[int],
) -> dict[str, torch.Tensor]:
    generated = torch.tensor([generated_ids], dtype=torch.long)
    full_input_ids = torch.cat([our_kwargs["input_ids"], generated], dim=1)
    full_attention = torch.ones_like(full_input_ids)
    our_full = dict(our_kwargs)
    our_full["input_ids"] = full_input_ids
    position_ids, _ = get_qwen3_vl_rope_index_from_config(
        full_input_ids,
        config=config,
        image_grid_thw=our_kwargs.get("image_grid_thw"),
        video_grid_thw=our_kwargs.get("video_grid_thw"),
        attention_mask=full_attention,
    )
    our_full["position_ids"] = position_ids

    with torch.inference_mode():
        hidden = our_model(**_to_cuda_kwargs(our_full))
        results = {}
        for precision in ("fp32", "model"):
            our_model.logits_precision = precision
            results[precision] = (
                our_model.compute_logits(hidden)[:, -MAX_TOKENS - 1 : -1, :].detach().cpu()
            )
        return results


def _compare_distribution(
    case_name: str, hf_logits: torch.Tensor, our_logits: torch.Tensor, generated_ids: list[int]
) -> None:
    hf_f = hf_logits.float()
    our_f = our_logits.float()
    diff = (hf_f - our_f).abs()
    targets = torch.tensor(generated_ids, dtype=torch.long).view(1, -1)
    hf_loss = torch.nn.functional.cross_entropy(
        hf_f.view(-1, hf_f.shape[-1]),
        targets.view(-1),
    )
    our_loss = torch.nn.functional.cross_entropy(
        our_f.view(-1, our_f.shape[-1]),
        targets.view(-1),
    )
    hf_ppl = torch.exp(hf_loss)
    our_ppl = torch.exp(our_loss)
    ppl_diff = (hf_ppl - our_ppl).abs()

    print(f"{case_name} logits shape HF: {list(hf_logits.shape)}")
    print(f"{case_name} logits shape Prism: {list(our_logits.shape)}")
    print(f"{case_name} generated token count: {len(generated_ids)}")
    print(f"{case_name} HF mean/std: {hf_f.mean().item():.6e} / {hf_f.std().item():.6e}")
    print(f"{case_name} Prism mean/std: {our_f.mean().item():.6e} / {our_f.std().item():.6e}")
    print(f"{case_name} logits max diff: {diff.max().item():.6e}")
    print(f"{case_name} logits mean diff: {diff.mean().item():.6e}")
    print(f"{case_name} HF loss/ppl: {hf_loss.item():.6e} / {hf_ppl.item():.6e}")
    print(f"{case_name} Prism loss/ppl: {our_loss.item():.6e} / {our_ppl.item():.6e}")
    print(f"{case_name} ppl diff: {ppl_diff.item():.6e}")

    assert list(hf_logits.shape) == [1, MAX_TOKENS, hf_logits.shape[-1]]
    assert list(our_logits.shape) == list(hf_logits.shape)
    assert diff.max().item() < MAX_LOGIT_DIFF_TOL
    assert diff.mean().item() < MEAN_LOGIT_DIFF_TOL
    assert ppl_diff.item() < PPL_DIFF_TOL
    print(f"{case_name} teacher-forced logits distribution: PASS")


def test_vl_teacher_forced_logits_distribution_and_ppl_match_hf() -> None:
    """生成轨迹上 32-token logits 分布和 perplexity 应与 HF 对齐。"""

    _require_cuda()
    transformers = require_transformers()
    model_path = get_model_path()
    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    hf_references = []
    hf_config = None
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    try:
        hf_config = hf_model.config
        for case in _cases():
            inputs, hf_kwargs, our_kwargs, visual_token_count = _prepare_case(
                processor, hf_config, case
            )
            with torch.inference_mode():
                hf_generated = hf_model.generate(
                    **_to_cuda_kwargs(with_hf_mm_token_type_ids(hf_model, hf_kwargs)),
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False,
                )
            generated_ids = hf_generated[0, inputs.input_ids.shape[1] :].tolist()
            for checkpoint in CHECKPOINTS:
                print(f"{case['name']} prefix@{checkpoint}: {generated_ids[:checkpoint]}")
            print(f"{case['name']} prompt tokens: {inputs.input_ids.shape[1]}")
            print(f"{case['name']} visual tokens: {visual_token_count}")

            hf_logits = _teacher_forced_hf_logits(hf_model, hf_kwargs, generated_ids)
            hf_references.append((case, our_kwargs, generated_ids, hf_logits))
            del hf_generated
            torch.cuda.empty_cache()
    finally:
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

    hf_cpu = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    our_model = _load_our_from_hf_cpu(hf_cpu)
    del hf_cpu
    gc.collect()
    torch.cuda.empty_cache()
    try:
        for case, our_kwargs, generated_ids, hf_logits in hf_references:
            logits_by_precision = _teacher_forced_our_logits(
                our_model,
                hf_config,
                our_kwargs,
                generated_ids,
            )
            for precision, our_logits in logits_by_precision.items():
                _compare_distribution(
                    f"{case['name']}[{precision}]",
                    hf_logits,
                    our_logits,
                    generated_ids,
                )
            del hf_logits, logits_by_precision
            torch.cuda.empty_cache()
    finally:
        del our_model
        gc.collect()
        torch.cuda.empty_cache()
