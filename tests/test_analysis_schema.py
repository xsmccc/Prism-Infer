"""P4 KV trace schema 与 visual span 定位验证。"""

import torch

from prism_infer.analysis.kv_trace import (
    build_trace_metadata,
    kv_trace,
    locate_token_spans,
)
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams


def test_locate_token_spans_groups_text_image_video_tokens():
    """visual token span 应按连续 image/video placeholder 分组。"""

    token_ids = [10, 11, 100, 100, 12, 200, 200, 200, 13]
    spans = locate_token_spans(token_ids, image_token_id=100, video_token_id=200)
    compact = [
        (span.modality, span.start, span.end, span.index, span.token_count)
        for span in spans
    ]

    print(f"token ids length: {len(token_ids)}")
    print(f"span compact: {compact}")

    assert compact == [
        ("text", 0, 2, 0, 2),
        ("image", 2, 4, 0, 2),
        ("text", 4, 5, 1, 1),
        ("video", 5, 8, 0, 3),
        ("text", 8, 9, 2, 1),
    ]
    print("KV trace span grouping: PASS")


def test_build_trace_metadata_serializes_shapes_and_visual_spans():
    """trace metadata 应包含输入 shape、grid、block table 和 flat span。"""

    seq = Sequence(
        [1, 10, 10, 2, 20, 20, 20, 3],
        SamplingParams(temperature=0.0, max_tokens=1),
        pixel_values=torch.zeros(4, 8),
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
        pixel_values_videos=torch.zeros(8, 8),
        video_grid_thw=torch.tensor([[2, 4, 4]], dtype=torch.long),
        image_token_id=10,
        image_token_count=2,
        video_token_id=20,
        video_token_count=3,
    )
    seq.block_table = [7]
    with kv_trace(metadata={"case": "schema"}):
        metadata = build_trace_metadata(
            [seq],
            is_prefill=True,
            input_ids=torch.arange(len(seq), dtype=torch.long),
            position_ids=torch.arange(len(seq), dtype=torch.long),
            slot_mapping=torch.arange(len(seq), dtype=torch.int32),
            block_tables=None,
            context_lens=None,
            block_size=256,
        )

    assert metadata is not None
    payload = metadata.to_dict()
    spans = payload["sequences"][0]["spans"]
    image_span = next(span for span in spans if span["modality"] == "image")
    video_span = next(span for span in spans if span["modality"] == "video")

    print(f"metadata input shape: {payload['input_ids_shape']}")
    print(f"metadata position shape: {payload['position_ids_shape']}")
    print(f"metadata image grid: {payload['sequences'][0]['image_grid_thw']}")
    print(f"metadata video grid: {payload['sequences'][0]['video_grid_thw']}")
    print(f"image span: {image_span}")
    print(f"video span: {video_span}")

    assert payload["phase"] == "prefill"
    assert payload["input_ids_shape"] == [8]
    assert payload["slot_mapping_shape"] == [8]
    assert payload["sequences"][0]["block_table"] == [7]
    assert payload["sequences"][0]["image_grid_thw"] == [[1, 4, 4]]
    assert payload["sequences"][0]["video_grid_thw"] == [[2, 4, 4]]
    assert image_span["flat_start"] == 1
    assert image_span["flat_end"] == 3
    assert video_span["flat_start"] == 4
    assert video_span["flat_end"] == 7
    print("KV trace metadata schema: PASS")
