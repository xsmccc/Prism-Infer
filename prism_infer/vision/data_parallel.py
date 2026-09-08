"""Whole-image encoder work sharing inside the existing tensor-parallel group."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist

from prism_infer.observability import profile_region

VisualEncoder = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, list[torch.Tensor]],
]


def assign_images_by_patch_count(
    patch_counts: Sequence[int],
    world_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Greedily balance whole images, retaining source order within each rank."""

    assigned: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for image_id in sorted(range(len(patch_counts)), key=lambda i: (-patch_counts[i], i)):
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        assigned[owner].append(image_id)
        loads[owner] += patch_counts[image_id]
    return tuple(tuple(sorted(image_ids)) for image_ids in assigned)


def encode_images_data_parallel(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    encode_local: VisualEncoder,
    *,
    spatial_merge_size: int,
    output_size: int,
    num_deepstack_features: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Encode image shards and return complete, source-ordered outputs on every rank.

    The caller owns the initialized TP process group and invokes this helper on
    every rank with the same nonempty image payload and validated grid. Video
    and TP1 dispatch remain outside this helper. ``encode_local`` is the
    unmodified local encoder, including its whole-segment microbatch policy.

    Planning reads the small grid once on the host. Collective sizes are derived
    from that shared grid, without object collectives or extra CUDA synchronizes.
    Empty ranks skip the encoder but participate in the single padded all-gather.
    """

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    grid_rows = [tuple(int(value) for value in row) for row in grid_thw.tolist()]
    patch_counts = [t * h * w for t, h, w in grid_rows]
    if sum(patch_counts) != pixel_values.shape[0]:
        raise ValueError(
            "image grid/payload patch counts do not match: "
            f"grid={sum(patch_counts)} payload={pixel_values.shape[0]}"
        )
    merge_area = spatial_merge_size**2
    merged_counts = [count // merge_area for count in patch_counts]
    assignments = assign_images_by_patch_count(patch_counts, world_size)
    rows_by_rank = [sum(merged_counts[index] for index in ids) for ids in assignments]
    total_rows = sum(merged_counts)
    components = 1 + num_deepstack_features
    patch_offsets = [0]
    merged_offsets = [0]
    for patch_count, merged_count in zip(patch_counts, merged_counts, strict=True):
        patch_offsets.append(patch_offsets[-1] + patch_count)
        merged_offsets.append(merged_offsets[-1] + merged_count)

    max_rows = max(rows_by_rank)
    packed = torch.zeros(
        (components, max_rows, output_size),
        device=pixel_values.device,
        dtype=dtype,
    )
    image_ids = assignments[rank]
    with profile_region(
        "model.vision.data_parallel.encode",
        cuda=pixel_values.is_cuda,
        metadata={
            "rank": rank,
            "images": len(image_ids),
            "patches": sum(patch_counts[index] for index in image_ids),
            "merged_rows": rows_by_rank[rank],
        },
    ):
        if image_ids:
            pixel_parts = [
                pixel_values[patch_offsets[index] : patch_offsets[index + 1]] for index in image_ids
            ]
            local_pixels = (
                pixel_parts[0] if len(pixel_parts) == 1 else torch.cat(pixel_parts, dim=0)
            )
            local_grid = torch.tensor(
                [grid_rows[index] for index in image_ids],
                device=grid_thw.device,
                dtype=grid_thw.dtype,
            )
            main, deepstack = encode_local(local_pixels, local_grid)
            expected_shape = (rows_by_rank[rank], output_size)
            if len(deepstack) != num_deepstack_features or any(
                value.shape != expected_shape for value in (main, *deepstack)
            ):
                raise RuntimeError(
                    f"rank {rank} image encoder expected main and {num_deepstack_features} "
                    f"DeepStack tensors of shape {expected_shape}, got main={tuple(main.shape)} "
                    f"DeepStack={[tuple(value.shape) for value in deepstack]}"
                )
            for component, value in enumerate((main, *deepstack)):
                packed[component, : rows_by_rank[rank]].copy_(value)
            del main, deepstack, value, local_pixels, local_grid, pixel_parts

    gathered = torch.empty(
        (world_size, components, max_rows, output_size),
        device=pixel_values.device,
        dtype=dtype,
    )
    with profile_region(
        "model.vision.data_parallel.gather",
        cuda=pixel_values.is_cuda,
        metadata={
            "world_size": world_size,
            "max_merged_rows_per_rank": max_rows,
            "packed_bytes_per_rank": packed.numel() * packed.element_size(),
        },
    ):
        dist.all_gather_into_tensor(gathered.view(-1), packed.view(-1))
    del packed

    with profile_region("model.vision.data_parallel.restore", cuda=pixel_values.is_cuda):
        outputs = torch.empty(
            (components, total_rows, output_size),
            device=pixel_values.device,
            dtype=dtype,
        )
        for owner, owner_image_ids in enumerate(assignments):
            source_row = 0
            for image_id in owner_image_ids:
                row_count = merged_counts[image_id]
                outputs[:, merged_offsets[image_id] : merged_offsets[image_id + 1]].copy_(
                    gathered[owner, :, source_row : source_row + row_count]
                )
                source_row += row_count

    return outputs[0], list(outputs[1:].unbind(0))
