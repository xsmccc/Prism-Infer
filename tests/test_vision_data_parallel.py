"""Whole-image assignment and real collective output-order checks, without a GPU."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prism_infer.vision.data_parallel import (
    assign_images_by_patch_count,
    encode_images_data_parallel,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("patch_counts", "world_size", "expected"),
    [
        ([8, 8, 8, 8], 2, ((0, 2), (1, 3))),
        ([4, 16, 8, 48, 4], 2, ((3,), (0, 1, 2, 4))),
        ([8, 16], 4, ((1,), (0,), (), ())),
        ([4], 2, ((0,), ())),
    ],
)
def test_assignment_is_balanced_deterministic_and_source_ordered(
    patch_counts,
    world_size,
    expected,
) -> None:
    assert assign_images_by_patch_count(patch_counts, world_size) == expected


def _check_distributed_image_outputs(
    rank: int,
    world_size: int,
    rendezvous: str,
    grid_rows: tuple[tuple[int, int, int], ...],
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        grid = torch.tensor(grid_rows, dtype=torch.int64)
        patch_counts = [t * h * w for t, h, w in grid_rows]
        output_size = 3
        pixels = torch.arange(sum(patch_counts), dtype=torch.float32).unsqueeze(1) + torch.arange(
            output_size, dtype=torch.float32
        ).unsqueeze(0)
        assignments = assign_images_by_patch_count(patch_counts, world_size)
        patch_offsets = [0]
        for count in patch_counts:
            patch_offsets.append(patch_offsets[-1] + count)
        calls = []

        def encode_local(local_pixels, local_grid):
            local_ids = assignments[rank]
            calls.append(local_ids)
            assert local_grid.tolist() == [list(grid_rows[index]) for index in local_ids]
            expected_pixels = torch.cat(
                [pixels[patch_offsets[index] : patch_offsets[index + 1]] for index in local_ids]
            )
            assert torch.equal(local_pixels, expected_pixels)
            # Keep a distinct marker for each spatially merged row, not just each image.
            main = local_pixels[::4].to(torch.bfloat16)
            return main, [main + 16 * (layer + 1) for layer in range(3)]

        main, deepstack = encode_images_data_parallel(
            pixels,
            grid,
            encode_local,
            spatial_merge_size=2,
            output_size=output_size,
            num_deepstack_features=3,
            dtype=torch.bfloat16,
        )

        assert calls == ([assignments[rank]] if assignments[rank] else [])
        expected_main = pixels[::4].to(torch.bfloat16)
        assert main.shape == expected_main.shape
        assert main.dtype == torch.bfloat16
        assert main.is_contiguous()
        assert torch.equal(main, expected_main)
        assert len(deepstack) == 3
        for layer, values in enumerate(deepstack):
            assert values.dtype == torch.bfloat16
            assert values.is_contiguous()
            assert torch.equal(values, expected_main + 16 * (layer + 1))
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="CPU collective checks require torch.distributed with Gloo",
)
@pytest.mark.parametrize(
    ("world_size", "grid_rows"),
    [
        (2, ((1, 2, 2), (1, 4, 4), (1, 2, 4), (1, 6, 8), (1, 2, 2))),
        (4, ((1, 2, 4), (1, 4, 4))),
        (2, ((1, 2, 2),)),
    ],
    ids=["uneven-and-reordered", "fewer-images-than-ranks", "single-image"],
)
def test_collective_restores_every_image_row_and_deepstack(
    tmp_path: Path,
    world_size: int,
    grid_rows: tuple[tuple[int, int, int], ...],
) -> None:
    rendezvous = (tmp_path / "image-dp-rendezvous").resolve().as_uri()
    mp.spawn(
        _check_distributed_image_outputs,
        args=(world_size, rendezvous, grid_rows),
        nprocs=world_size,
        join=True,
    )
