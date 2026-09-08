"""Native FlashInfer Graph replay with unequal page counts and a padded bucket."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from prism_infer.ops.flashinfer_paged import FlashInferPagedDecode


def reference(q, keys, values, lengths, tables):
    outputs = []
    for row, length in enumerate(lengths.tolist()):
        ids = tables[row, : (length + 255) // 256].long()
        k = keys[ids].reshape(-1, 2, 128)[:length].transpose(0, 1).unsqueeze(0)
        v = values[ids].reshape(-1, 2, 128)[:length].transpose(0, 1).unsqueeze(0)
        outputs.append(
            F.scaled_dot_product_attention(
                q[row].unsqueeze(0).unsqueeze(2),
                k,
                v,
                enable_gqa=True,
            )
            .squeeze(0)
            .squeeze(1)
        )
    return torch.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(91)
    q = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(8, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    adapter = FlashInferPagedDecode(
        block_size=256,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,
        dtype=q.dtype,
        use_cuda_graph=True,
        max_batch=2,
        max_blocks=3,
    )
    cases = [
        ([256, 640], [[3, -1, -1], [4, 0, 2]]),
        ([257, 650], [[3, 1, -1], [4, 0, 2]]),
        ([257], [[3, 1]]),
    ]
    lengths = torch.tensor(cases[0][0], device="cuda", dtype=torch.int32)
    tables = torch.tensor(cases[0][1], device="cuda", dtype=torch.int32)
    adapter.stage_plan_inputs(lengths, tables)
    for _ in range(2):
        adapter.run(q, keys, values)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = adapter.run(q, keys, values)
    report = []
    for length_values, table_values in cases:
        lengths = torch.tensor(length_values, device="cuda", dtype=torch.int32)
        tables = torch.tensor(table_values, device="cuda", dtype=torch.int32)
        expected = reference(q, keys, values, lengths, tables)
        adapter.stage_plan_inputs(lengths, tables)
        graph.replay()
        actual = output[: len(length_values)]
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        report.append(
            {
                "lengths": length_values,
                "tables": table_values,
                "max_abs_error": float((actual.float() - expected.float()).abs().max()),
            }
        )
    args.output.write_text(
        json.dumps({"scope": "native BF16 FlashInfer numerical check", "cases": report}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
