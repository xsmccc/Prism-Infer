#!/usr/bin/env python3
"""Validate and compare a Prism P9 artifact with a vLLM quality artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.p9_external_quality import compare_prism_external_quality
from prism_infer.analysis.p9_quality_materialization import write_json_atomic
from prism_infer.analysis.p9_quality_runtime import (
    load_reference_records_for_artifacts,
    read_json_object,
)

DEFAULT_EVALUATOR = REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json"
DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"
DEFAULT_MATERIALIZED_ROOT = REPO_ROOT / "data/p9_quality/materialized"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prism", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=DEFAULT_MATERIALIZED_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-headline", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite comparison artifact: {args.output}")

    prism = read_json_object(args.prism)
    external = read_json_object(args.external)
    evaluator = read_json_object(args.evaluator)
    protocol = read_json_object(args.protocol)
    reference_records = load_reference_records_for_artifacts(
        args.materialized_root,
        [prism, external],
        protocol=protocol,
    )
    model_path = external.get("run_contract", {}).get("model")
    if not isinstance(model_path, str):
        raise ValueError("external artifact has no model path")
    model_config = read_json_object(Path(model_path) / "config.json")
    result = compare_prism_external_quality(
        prism,
        external,
        evaluator=evaluator,
        protocol=protocol,
        model_config=model_config,
        reference_records=reference_records,
        require_headline=args.require_headline,
    )
    output_sha256 = write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "dataset": result["dataset"],
                "external_mode": result["external_mode"],
                "samples": result["samples"],
                "decision": result["decision"],
                "semantic_input_exact": result["semantic_input_exact"],
                "all_required_metrics_pass": result["all_required_metrics_pass"],
                "full_physical_comparable": result["kv_cache"][
                    "full_physical_comparable"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["decision"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
