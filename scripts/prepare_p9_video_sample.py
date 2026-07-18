#!/usr/bin/env python3
"""Decode one frozen MVBench sample and emit a lossless temporary RGB bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer.analysis.p9_quality_runtime import (
    close_images,
    read_json_object,
    read_jsonl_objects,
    safe_materialized_path,
)
from prism_infer.analysis.p9_video_bundle import write_video_bundle
from prism_infer.analysis.p9_video_sampling import (
    sample_frame_manifest,
    sample_video_file,
)

DEFAULT_EVALUATOR = REPO_ROOT / "benchmarks/workloads/p9_quality_evaluator.json"
DEFAULT_MATERIALIZED_ROOT = REPO_ROOT / "data/p9_quality/materialized"
DEFAULT_RECORDS = DEFAULT_MATERIALIZED_ROOT / "records/mvbench_test.final.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=DEFAULT_MATERIALIZED_ROOT,
    )
    args = parser.parse_args()

    evaluator = read_json_object(args.evaluator)
    dataset_evaluator = evaluator["datasets"]["mvbench_test"]
    runtime = evaluator["runtime"]
    records = read_jsonl_objects(args.records)
    matches = [record for record in records if record["sample_id"] == args.sample_id]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one MVBench record for {args.sample_id!r}, "
            f"found {len(matches)}"
        )
    record = matches[0]
    media = record["media"][0]
    source_sha256 = media.get("sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise SystemExit("MVBench record does not have a resolved source identity")

    frames = []
    try:
        if media.get("identity_kind") == "canonical_frame_manifest_sha256":
            frames, video_sampling = sample_frame_manifest(
                media["frames"],
                materialized_root=args.materialized_root,
                frames=runtime["video_frames"],
                fps=dataset_evaluator["video_sampling"]["frame_directory_fps"],
                temporal_bound=record["temporal_bound"],
            )
        else:
            source = safe_materialized_path(
                args.materialized_root,
                media["materialized_path"],
            )
            frames, video_sampling = sample_video_file(
                source,
                frames=runtime["video_frames"],
                temporal_bound=record["temporal_bound"],
                decoder_contract=dataset_evaluator["video_sampling"][
                    "video_file_decoder"
                ],
            )
        evidence = write_video_bundle(
            args.output,
            frames,
            sample_id=record["sample_id"],
            source_media_sha256=source_sha256,
            video_sampling=video_sampling,
        )
    finally:
        close_images(frames)
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
