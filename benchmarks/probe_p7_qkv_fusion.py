"""Probe whether a packed QKV projection preserves separate BF16 outputs。

P7.5 uses this correctness-only probe before considering QKV packing. It deliberately
does not report timing: a candidate that changes strict BF16 outputs is rejected before
performance measurement, and a busy external GPU cannot turn this into a speed claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_packed_mlp import _git_metadata, _gpu_baseline


def _difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = (actual - expected).abs()
    return {
        "exact": torch.equal(actual, expected),
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    batches = [int(value) for value in args.batch_sizes.split(",") if value]
    if not batches or any(batch < 1 for batch in batches):
        raise SystemExit("batch sizes must be positive")

    q_size = args.num_heads * args.head_dim
    kv_size = args.num_kv_heads * args.head_dim
    if q_size != args.hidden_size:
        raise SystemExit("this Qwen probe expects num_heads * head_dim == hidden_size")

    commit, dirty = _git_metadata()
    baseline = _gpu_baseline()
    torch.manual_seed(20260717)
    weights = [
        torch.randn(
            output_size,
            args.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for output_size in (q_size, kv_size, kv_size)
    ]
    packed_weight = torch.cat(weights, dim=0)
    cases = []
    with torch.inference_mode():
        for batch in batches:
            x = torch.randn(
                batch,
                args.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            separate = tuple(F.linear(x, weight) for weight in weights)
            packed = F.linear(x, packed_weight).split(
                (q_size, kv_size, kv_size),
                dim=-1,
            )
            components = {
                name: _difference(actual, expected)
                for name, actual, expected in zip(
                    ("q", "k", "v"),
                    packed,
                    separate,
                )
            }
            cases.append(
                {
                    "batch_size": batch,
                    "all_exact": all(
                        component["exact"]
                        for component in components.values()
                    ),
                    "components": components,
                }
            )

    record = {
        "schema_version": 1,
        "record_type": "p75_qkv_fusion_correctness_probe",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "git_commit": commit,
            "git_dirty": dirty,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_baseline": baseline,
        },
        "configuration": {
            "dtype": "torch.bfloat16",
            "hidden_size": args.hidden_size,
            "q_size": q_size,
            "kv_size": kv_size,
            "batch_sizes": batches,
            "seed": 20260717,
        },
        "all_exact": all(case["all_exact"] for case in cases),
        "candidate_status": (
            "eligible_for_performance_probe"
            if all(case["all_exact"] for case in cases)
            else "rejected_by_strict_correctness"
        ),
        "performance_measured": False,
        "cases": cases,
        "claim_boundaries": [
            "The probe checks isolated BF16 projection outputs, not full-model quality.",
            "No timing is reported because strict correctness is evaluated first.",
            "External GPU activity does not make this record performance evidence.",
        ],
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
