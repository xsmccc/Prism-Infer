"""Compare distributed image features against the original full Vision Encoder.

Run with torchrun --standalone --nproc_per_node=2 on an allocated GPU node.
This checks feature assembly and floating-point drift, not task accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import AutoConfig

from prism_infer.models.qwen3_vl import Qwen3VLModel
from prism_infer.vision.data_parallel import encode_images_data_parallel
from prism_infer.vision.vision_encoder import VisionEncoder


def _feature_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    actual = actual.float()
    difference = actual - reference
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (difference.norm() / reference.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            reference.flatten(), actual.flatten(), dim=0
        ).item(),
        "equal_fraction": (actual == reference).float().mean().item(),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    visual = VisionEncoder(config.vision_config, dtype=torch.bfloat16).to(device).eval()
    prefix = "model.visual."
    for shard in Path(args.model).glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name.startswith(prefix):
                    visual.get_parameter(name[len(prefix) :]).copy_(weights.get_tensor(name))

    def encode_local(pixels: torch.Tensor, grid: torch.Tensor):
        if pixels.shape[0] <= 4096:
            return visual(pixels, grid)
        results = []
        for start, end, rows in Qwen3VLModel._plan_visual_microbatches(grid, patch_limit=4096):
            results.append(visual(pixels[start:end], grid.new_tensor(rows)))
        return (
            torch.cat([result[0] for result in results]),
            [
                torch.cat([result[1][layer] for result in results])
                for layer in range(len(visual.deepstack_visual_indexes))
            ],
        )

    cases = {
        "eight_equal_images": [[1, 28, 28]] * 8,
        "unequal_images": [[1, 16, 24], [1, 32, 48], [1, 28, 28]],
        "single_image_empty_peer": [[1, 28, 28]],
    }
    results = []
    generator = torch.Generator(device="cpu").manual_seed(20260908)
    patch_width = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size**2
    )
    for name, rows in cases.items():
        grid = torch.tensor(rows, dtype=torch.int64, device=device)
        patches = sum(t * h * w for t, h, w in rows)
        pixels = torch.randn(patches, patch_width, generator=generator).to(device)

        def encode_distributed(pixels=pixels, grid=grid):
            return encode_images_data_parallel(
                pixels,
                grid,
                encode_local,
                spatial_merge_size=visual.spatial_merge_size,
                output_size=config.vision_config.out_hidden_size,
                num_deepstack_features=len(visual.deepstack_visual_indexes),
                dtype=torch.bfloat16,
            )

        reference = encode_local(pixels, grid)
        distributed = encode_distributed()
        torch.cuda.synchronize()
        errors = [
            _feature_error(expected, actual)
            for expected, actual in zip(
                (reference[0], *reference[1]), (distributed[0], *distributed[1]), strict=True
            )
        ]
        times = {"replicated_ms": [], "data_ms": []}
        for _ in range(3):
            for label, call in (
                ("replicated_ms", partial(encode_local, pixels, grid)),
                ("data_ms", encode_distributed),
            ):
                dist.barrier()
                torch.cuda.synchronize()
                start = perf_counter()
                call()
                torch.cuda.synchronize()
                times[label].append((perf_counter() - start) * 1000)
        results.append({"case": name, "grid": rows, "features": errors, **times})
        print(json.dumps({"rank": rank, **results[-1]}), flush=True)
    all_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_results, {"rank": rank, "cases": results})
    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"model": args.model, "ranks": all_results}, indent=2) + "\n",
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
