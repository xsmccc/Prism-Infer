#!/usr/bin/env python3
"""Validate and compare paired P9 baseline/candidate quality artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.benchmark_schema import canonical_json_sha256
from prism_infer.analysis.p9_quality_comparison import compare_quality_artifacts
from prism_infer.analysis.p9_quality_materialization import (
    sha256_file,
    write_json_atomic,
)


DEFAULT_EVALUATOR = REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json"
DEFAULT_PROTOCOL = REPO_ROOT / "benchmarks/workloads/p9_quality_protocol.json"
DEFAULT_MATERIALIZED_ROOT = REPO_ROOT / "data/p9_quality/materialized"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"expected non-empty JSONL records: {path}")
    return records


def _load_reference_records(
    materialized_root: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    root = materialized_root.resolve()
    manifest_path = root / "p9_quality_materialization.json"
    manifest_sha256 = sha256_file(manifest_path)
    for label, artifact in (("baseline", baseline), ("candidate", candidate)):
        expected = artifact.get("run_contract", {}).get(
            "materialization_manifest_sha256"
        )
        if expected != manifest_sha256:
            raise ValueError(
                f"{label} artifact references a different materialization manifest"
            )

    baseline_dataset = baseline.get("run_contract", {}).get("dataset")
    candidate_dataset = candidate.get("run_contract", {}).get("dataset")
    if baseline_dataset != candidate_dataset or not isinstance(baseline_dataset, str):
        raise ValueError("quality artifacts do not reference the same dataset")

    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("materialization manifest has unsupported schema_version")
    if manifest.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("materialization manifest references a different protocol")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("materialization manifest has no dataset list")
    matches = [
        artifact
        for artifact in datasets
        if isinstance(artifact, dict) and artifact.get("id") == baseline_dataset
    ]
    if len(matches) != 1:
        raise ValueError(
            f"materialization manifest must contain dataset {baseline_dataset!r} once"
        )
    selected = matches[0].get("selected_records")
    if not isinstance(selected, dict):
        raise ValueError("materialization dataset has no selected_records identity")
    relative_path = selected.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("materialization selected_records.path is invalid")
    records_path = (root / relative_path).resolve()
    if not records_path.is_relative_to(root) or not records_path.is_file():
        raise ValueError("materialization selected_records.path escapes its root")
    if sha256_file(records_path) != selected.get("sha256"):
        raise ValueError("materialized selected records SHA256 mismatch")
    return _read_jsonl(records_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=DEFAULT_MATERIALIZED_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-headline",
        action="store_true",
        help="Reject smoke artifacts and require clean formal runs.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite comparison artifact: {args.output}")

    baseline = _read_object(args.baseline)
    candidate = _read_object(args.candidate)
    evaluator = _read_object(args.evaluator)
    protocol = _read_object(args.protocol)
    reference_records = _load_reference_records(
        args.materialized_root,
        baseline,
        candidate,
        protocol=protocol,
    )
    result = compare_quality_artifacts(
        baseline,
        candidate,
        evaluator=evaluator,
        protocol=protocol,
        require_headline=args.require_headline,
        reference_records=reference_records,
    )
    output_sha256 = write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "dataset": result["dataset"],
                "candidate_mode": result["candidate_mode"],
                "samples": result["samples"],
                "decision": result["decision"],
                "all_required_metrics_pass": result["all_required_metrics_pass"],
                "headline_eligible": result["headline_eligible"],
                "reference_scores_recomputed": result["reference_scores_recomputed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["decision"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
