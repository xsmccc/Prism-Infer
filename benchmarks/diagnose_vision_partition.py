"""Locate Vision partition drift on one GPU, without distributed communication.

Uses the checkpoint loader, seed, case order, and 4096-patch local microbatch
policy from check_vision_parallel.py. This is a numerical diagnostic, not a
latency benchmark: stage snapshots are copied to CPU for comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from prism_infer.models.qwen3_vl import Qwen3VLModel
from prism_infer.vision.data_parallel import assign_images_by_patch_count
from prism_infer.vision.vision_encoder import VisionEncoder

CASES = {
    "eight_equal_images": [[1, 28, 28]] * 8,
    "unequal_images": [[1, 16, 24], [1, 32, 48], [1, 28, 28]],
    "single_image_empty_peer": [[1, 28, 28]],
}
StageRecorder = Callable[[str, torch.Tensor, bool, int], None]


def _trace_encoder(
    visual: VisionEncoder,
    pixels: torch.Tensor,
    grid: torch.Tensor,
    record: StageRecorder,
) -> None:
    """Execute the existing preparation/block/merger operations, recording outputs."""

    patch_hook = visual.patch_embed.register_forward_hook(
        lambda module, inputs, output: record("patch_embed", output, False, 0)
    )
    try:
        x, cos, sin, cu_seqlens, max_seqlen, ranges = visual.prepare_tensor_region_inputs(
            pixels, grid
        )
    finally:
        patch_hook.remove()
    # Re-evaluate this stateless geometry operation to separate its output from
    # the BF16 addition. Subtracting patch_embed from x would add rounding error.
    record("position_embedding", visual._pos_embed_interpolate(grid), False, 0)
    record("position_plus_patch", x, False, 0)
    record("rope_cos", cos, False, 0)
    record("rope_sin", sin, False, 0)

    first_block = visual.blocks[0]
    detail_modules = (
        ("block00.norm1", first_block.norm1, 0),
        ("block00.qkv", first_block.attn.qkv, 1),
        ("block00.attention_projection", first_block.attn.proj, 1),
        ("block00.attention", first_block.attn, 0),
        ("block00.norm2", first_block.norm2, 0),
        ("block00.mlp_fc1", first_block.mlp.linear_fc1, 0),
        ("block00.mlp_fc2", first_block.mlp.linear_fc2, 0),
    )
    hooks = []
    for name, module, row_axis in detail_modules:

        def hook(module, inputs, output, name=name, row_axis=row_axis):
            record(name, output, False, row_axis)

        hooks.append(module.register_forward_hook(hook))
    try:
        for block_index, block in enumerate(visual.blocks):
            x = block(
                x,
                cos=cos,
                sin=sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                segment_ranges=ranges,
            )
            record(f"block{block_index:02d}", x, False, 0)
            if block_index in visual.deepstack_visual_indexes:
                merger_index = visual.deepstack_visual_indexes.index(block_index)
                deepstack = visual.deepstack_merger_list[merger_index](x)
                record(f"deepstack{merger_index}.block{block_index:02d}", deepstack, True, 0)
        record("main_merger", visual.merger(x), True, 0)
    finally:
        for handle in hooks:
            handle.remove()


def _error_totals(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | int]:
    reference = reference.float()
    actual = actual.float()
    difference = actual - reference
    return {
        "elements": reference.numel(),
        "unequal_elements": int((difference != 0).sum().item()),
        "nonfinite_elements": int((~torch.isfinite(actual)).sum().item()),
        "max_abs": difference.abs().max().item(),
        "sum_abs": difference.abs().sum(dtype=torch.float64).item(),
        "sum_squared_error": difference.double().square().sum().item(),
        "sum_squared_reference": reference.double().square().sum().item(),
    }


def _finish_error(totals: dict[str, float | int]) -> dict[str, float | int | None]:
    reference_squared = totals["sum_squared_reference"]
    return {
        "elements": totals["elements"],
        "unequal_elements": totals["unequal_elements"],
        "nonfinite_elements": totals["nonfinite_elements"],
        "equal_fraction": 1 - totals["unequal_elements"] / totals["elements"],
        "max_abs": totals["max_abs"],
        "mean_abs": totals["sum_abs"] / totals["elements"],
        "relative_l2": (
            math.sqrt(totals["sum_squared_error"] / reference_squared)
            if reference_squared
            else None
        ),
    }


def _run_case(
    visual: VisionEncoder,
    pixels: torch.Tensor,
    rows: list[list[int]],
    shards: int,
) -> dict:
    merge_area = visual.spatial_merge_size**2
    patch_counts = [t * h * w for t, h, w in rows]
    patch_offsets = [0]
    for count in patch_counts:
        patch_offsets.append(patch_offsets[-1] + count)
    total_patches = patch_offsets[-1]
    references: dict[str, torch.Tensor] = {}
    aggregate_errors: dict[str, dict] = {}
    partition_records = []

    def run_partition(image_ids: tuple[int, ...], *, reference: bool) -> dict:
        if not image_ids:
            return {"images": [], "skipped_encoder": True, "microbatches": []}
        pieces = [pixels[patch_offsets[i] : patch_offsets[i + 1]] for i in image_ids]
        local_pixels = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
        grid = torch.tensor([rows[i] for i in image_ids], device=pixels.device, dtype=torch.int64)
        patch_ids = torch.cat(
            [torch.arange(patch_offsets[i], patch_offsets[i + 1]) for i in image_ids]
        )
        merged_ids = torch.cat(
            [
                torch.arange(patch_offsets[i] // merge_area, patch_offsets[i + 1] // merge_area)
                for i in image_ids
            ]
        )
        if local_pixels.shape[0] <= 4096:
            plans = ((0, int(local_pixels.shape[0]), tuple(tuple(row) for row in grid.tolist())),)
        else:
            plans = Qwen3VLModel._plan_visual_microbatches(grid, patch_limit=4096)
        microbatches = []
        for start, end, chunk_rows in plans:
            errors = {}
            raw_ids = patch_ids[start:end]
            output_ids = merged_ids[start // merge_area : end // merge_area]

            def record(
                name,
                tensor,
                merged,
                row_axis,
                output_ids=output_ids,
                raw_ids=raw_ids,
                errors=errors,
            ):
                values = tensor.detach().movedim(row_axis, 0).cpu()
                row_ids = output_ids if merged else raw_ids
                if reference:
                    if name not in references:
                        count = total_patches // merge_area if merged else total_patches
                        references[name] = torch.empty(
                            (count, *values.shape[1:]), dtype=values.dtype, device="cpu"
                        )
                    references[name].index_copy_(0, row_ids, values)
                    return
                expected = references[name].index_select(0, row_ids)
                totals = _error_totals(expected, values)
                errors[name] = _finish_error(totals)
                if name not in aggregate_errors:
                    aggregate_errors[name] = totals
                else:
                    accumulated = aggregate_errors[name]
                    for key, value in totals.items():
                        accumulated[key] = (
                            max(accumulated[key], value)
                            if key == "max_abs"
                            else accumulated[key] + value
                        )

            _trace_encoder(visual, local_pixels[start:end], grid.new_tensor(chunk_rows), record)
            microbatches.append(
                {"patch_range": [start, end], "grid": chunk_rows, "stage_errors": errors}
            )
        return {"images": image_ids, "skipped_encoder": False, "microbatches": microbatches}

    full = run_partition(tuple(range(len(rows))), reference=True)
    assignments = assign_images_by_patch_count(patch_counts, shards)
    for rank, image_ids in enumerate(assignments):
        partition_records.append({"rank": rank, **run_partition(image_ids, reference=False)})
    stage_errors = {name: _finish_error(aggregate_errors[name]) for name in references}
    return {
        "grid": rows,
        "assignments": assignments,
        "full_microbatches": full["microbatches"],
        "first_unequal_stage": next(
            (name for name, error in stage_errors.items() if error["unequal_elements"]), None
        ),
        "stage_errors": stage_errors,
        "partitions": partition_records,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    visual = VisionEncoder(config.vision_config, dtype=torch.bfloat16).to(device).eval()
    loaded_parameters = 0
    for shard in Path(args.model).glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name.startswith("model.visual."):
                    visual.get_parameter(name[len("model.visual.") :]).copy_(
                        weights.get_tensor(name)
                    )
                    loaded_parameters += 1

    patch_width = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size**2
    )
    generator = torch.Generator(device="cpu").manual_seed(20260908)
    inputs = {}
    for name, rows in CASES.items():
        patches = sum(t * h * w for t, h, w in rows)
        # Draw even unselected cases so the second/third inputs exactly retain
        # check_vision_parallel.py's CPU generator state and randn shapes.
        pixels = torch.randn(patches, patch_width, generator=generator)
        if name in args.cases:
            inputs[name] = pixels

    results = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "loaded_vision_parameters": loaded_parameters,
        "seed": 20260908,
        "rng_case_order": list(CASES),
        "communication": "none; sequential local shards on the same VisionEncoder",
        "microbatch_patch_limit": 4096,
        "fp32_weight_source": "loaded BF16 weights promoted to FP32",
        "comparison": "each dtype compares full versus partitioned; no tolerance gate",
        "runs": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for precision in ("bf16", "fp32_no_tf32"):
        if precision == "fp32_no_tf32":
            visual.float()
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        for name, cpu_pixels in inputs.items():
            result = {
                "precision": precision,
                "case": name,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "bf16_reduced_precision_reduction": (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                ),
                **_run_case(visual, cpu_pixels.to(device), CASES[name], args.shards),
            }
            results["runs"].append(result)
            output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "precision": precision,
                        "case": name,
                        "first_unequal_stage": result["first_unequal_stage"],
                        "stage_errors": result["stage_errors"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
