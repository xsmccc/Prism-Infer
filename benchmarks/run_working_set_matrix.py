#!/usr/bin/env python3
"""Run the fixed 15-cell repeated-visual-context performance matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import collect_git_metadata
from prism_infer.analysis.p9_quality_materialization import write_json_atomic
from prism_infer.analysis.working_set_plan import load_working_set_plan

WORKSETS = ("fit", "knee", "pressure")
PRISM_VARIANTS = ("vision_only", "dense_prefix", "compact_prefix")


@dataclass(frozen=True, slots=True)
class MatrixCell:
    engine: str
    workset: str
    variant: str

    @property
    def name(self) -> str:
        return f"{self.engine}_{self.variant}_{self.workset}"


def _cells() -> tuple[MatrixCell, ...]:
    cells = []
    for workset in WORKSETS:
        cells.extend(MatrixCell("prism", workset, variant) for variant in PRISM_VARIANTS)
        cells.append(MatrixCell("vllm", workset, "engine_default"))
        cells.append(MatrixCell("sglang", workset, "engine_default"))
    return tuple(cells)


def _command(
    cell: MatrixCell,
    *,
    model: Path,
    plan: Path,
    materialized_root: Path,
    output: Path,
    prism_python: Path,
    vllm_python: Path,
) -> list[str]:
    common = [
        "--model",
        str(model),
        "--working-set-plan",
        str(plan),
        "--working-set-id",
        cell.workset,
        "--materialized-root",
        str(materialized_root),
        "--output",
        str(output),
    ]
    if cell.engine == "prism":
        return [
            str(prism_python),
            str(REPO_ROOT / "benchmarks/bench_online.py"),
            *common,
            "--working-set-variant",
            cell.variant,
        ]
    script = REPO_ROOT / f"benchmarks/bench_online_{cell.engine}.py"
    return [str(vllm_python), str(script), *common, "--formal"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prism-python", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--sglang-overlay", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    git = collect_git_metadata(REPO_ROOT, strict=True)
    if git.dirty:
        raise SystemExit("formal matrix requires a clean Git worktree")
    plan = load_working_set_plan(args.plan)
    if {str(item["id"]) for item in plan["worksets"]} != set(WORKSETS):
        raise SystemExit("working-set plan must contain fit, knee and pressure")
    for path, label in (
        (args.model, "model"),
        (args.materialized_root, "materialized root"),
        (args.prism_python, "Prism Python"),
        (args.vllm_python, "vLLM Python"),
        (args.sglang_overlay, "SGLang overlay"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    progress_path = output_dir / "matrix_progress.json"
    if progress_path.exists():
        raise SystemExit(f"matrix progress already exists: {progress_path}")
    cells = _cells()
    outputs = {cell.name: raw_dir / f"{cell.name}.json" for cell in cells}
    logs = {cell.name: log_dir / f"{cell.name}.log" for cell in cells}
    collisions = [str(path) for path in (*outputs.values(), *logs.values()) if path.exists()]
    if collisions:
        raise SystemExit("refusing to replace matrix artifacts: " + ", ".join(collisions))
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    progress = {
        "record_type": "working_set_matrix_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": asdict(git),
        "plan": {
            "path": str(args.plan.resolve()),
            "sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        },
        "cells": [],
    }
    write_json_atomic(progress_path, progress)

    for index, cell in enumerate(cells, start=1):
        output = outputs[cell.name]
        log = logs[cell.name]
        command = _command(
            cell,
            model=args.model.resolve(),
            plan=args.plan.resolve(),
            materialized_root=args.materialized_root.resolve(),
            output=output,
            prism_python=args.prism_python.resolve(),
            vllm_python=args.vllm_python.resolve(),
        )
        record = {
            "index": index,
            **asdict(cell),
            "name": cell.name,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "output": str(output),
            "log": str(log),
            "command": command,
        }
        progress["cells"].append(record)
        write_json_atomic(progress_path, progress)
        environment = os.environ.copy()
        if cell.engine == "vllm":
            environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        elif cell.engine == "sglang":
            previous = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = str(args.sglang_overlay.resolve()) + (
                f"{os.pathsep}{previous}" if previous else ""
            )
        try:
            with log.open("x", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"matrix cell {cell.name} failed with exit code {completed.returncode}"
                )
            if not output.is_file():
                raise RuntimeError(f"matrix cell {cell.name} produced no output")
        except BaseException as exc:
            record.update(
                {
                    "status": "failed",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            progress["status"] = "failed"
            write_json_atomic(progress_path, progress)
            raise
        record.update(
            {
                "status": "complete",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
        write_json_atomic(progress_path, progress)

    progress["status"] = "complete"
    progress["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(progress_path, progress)
    print(json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
