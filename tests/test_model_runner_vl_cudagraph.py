"""P3.5 ModelRunner CUDA Graph decode shape 验证。"""

import torch

from prism_infer.engine.model_runner import ModelRunner


def test_cudagraph_decode_positions_normalize_text_and_vl_shapes() -> None:
    """graph replay 前 text/VL decode positions 都应规范为 `[3,batch]`。"""

    text_positions = torch.tensor([5, 7], dtype=torch.long)
    normalized_text = ModelRunner._as_mrope_decode_positions(text_positions)
    vl_positions = torch.tensor(
        [[5, 28, 56], [5, 28, 56], [5, 28, 56]],
        dtype=torch.long,
    )
    normalized_vl = ModelRunner._as_mrope_decode_positions(vl_positions)

    print(f"text decode input positions shape: {list(text_positions.shape)}")
    print(f"text graph positions shape: {list(normalized_text.shape)}")
    print(f"text graph positions: {normalized_text.tolist()}")
    print(f"vl decode input positions shape: {list(vl_positions.shape)}")
    print(f"vl graph positions shape: {list(normalized_vl.shape)}")
    print(f"vl graph positions: {normalized_vl.tolist()}")

    assert list(normalized_text.shape) == [3, 2]
    assert normalized_text.tolist() == [[5, 7], [5, 7], [5, 7]]
    assert list(normalized_vl.shape) == [3, 3]
    assert torch.equal(normalized_vl, vl_positions)
    print("ModelRunner CUDA Graph decode position shape normalization: PASS")


def test_cudagraph_batch_sizes_cover_non_standard_max_bs() -> None:
    """非 1/2/4/8/16 档位的 max_num_seqs 也必须有可 replay graph。"""

    cases = {
        1: [1],
        3: [1, 2, 3],
        5: [1, 2, 4, 5],
        17: [1, 2, 4, 8, 16, 17],
    }
    for max_bs, expected in cases.items():
        got = ModelRunner._cudagraph_batch_sizes(max_bs)
        print(f"max_bs={max_bs}, graph_bs={got}")
        assert got == expected
    print("ModelRunner CUDA Graph batch size coverage: PASS")
