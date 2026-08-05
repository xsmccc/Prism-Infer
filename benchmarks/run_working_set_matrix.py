#!/usr/bin/env python3
"""Run the repeated-visual-context performance comparison."""

from __future__ import annotations

import argparse
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
from prism_infer.analysis.identity import sha256_file
from prism_infer.analysis.quality_materialization import write_json_atomic
from prism_infer.analysis.working_set_plan import load_working_set_plan

WORKSETS = ("fit", "knee", "pressure")
PRISM_VARIANTS = ("vision_only", "dense_prefix", "compact_prefix")
PROGRESS_RECORD_TYPE = "working_set_matrix_progress"


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
    return [str(vllm_python), str(script), *common]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prism-python", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--sglang-overlay", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue after validating completed cells and archiving an interrupted attempt",
    )
    return parser.parse_args()


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    """Keep virtual-environment launchers intact while making paths absolute."""

    return Path(os.path.abspath(path))


def _validate_record_identity(
    record: object,
    *,
    cell: MatrixCell,
    index: int,
    output: Path,
    log: Path,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError(f"matrix progress cell {index} must be an object")
    expected = {
        "index": index,
        "engine": cell.engine,
        "workset": cell.workset,
        "variant": cell.variant,
        "name": cell.name,
        "output": str(output),
        "log": str(log),
    }
    mismatches = {
        key: {"expected": value, "actual": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(f"matrix progress cell {index} identity mismatch: {mismatches}")
    return record


def _validate_completed_cell(record: dict[str, object], *, output: Path, log: Path) -> None:
    if not output.is_file() or not log.is_file():
        raise ValueError(f"completed matrix cell is missing output or log: {output}")
    expected_sha256 = record.get("output_sha256")
    if not isinstance(expected_sha256, str) or sha256_file(output) != expected_sha256:
        raise ValueError(f"completed matrix cell output SHA256 mismatch: {output}")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"completed matrix cell output is not valid JSON: {output}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"completed matrix cell output must be a JSON object: {output}")


def _archive_interrupted_attempt(
    progress: dict[str, object],
    record: dict[str, object],
    *,
    output: Path,
    log: Path,
) -> None:
    attempts = progress.setdefault("interrupted_attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("matrix interrupted_attempts must be a list")
    attempt = len(attempts) + 1

    def archive(path: Path) -> str | None:
        if not path.exists():
            return None
        archived = path.with_name(f"{path.stem}.interrupted-{attempt}{path.suffix}")
        if archived.exists():
            raise FileExistsError(f"interrupted matrix artifact already exists: {archived}")
        path.rename(archived)
        return str(archived)

    attempts.append(
        {
            **record,
            "original_status": record.get("status"),
            "status": "interrupted",
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "archived_output": archive(output),
            "archived_log": archive(log),
        }
    )


def _resume_progress(
    progress_path: Path,
    *,
    cells: tuple[MatrixCell, ...],
    outputs: dict[str, Path],
    logs: dict[str, Path],
    git_record: dict[str, object],
    plan_record: dict[str, str],
) -> tuple[dict[str, object], int]:
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read matrix progress: {progress_path}") from exc
    if not isinstance(progress, dict) or progress.get("record_type") != PROGRESS_RECORD_TYPE:
        raise ValueError("matrix progress has an invalid record type")
    if progress.get("git") != git_record:
        raise ValueError("matrix progress Git identity differs from the current checkout")
    if progress.get("plan") != plan_record:
        raise ValueError("matrix progress plan identity differs from the requested plan")
    records = progress.get("cells")
    if not isinstance(records, list) or len(records) > len(cells):
        raise ValueError("matrix progress has an invalid cell list")

    completed = 0
    for offset, raw_record in enumerate(records):
        cell = cells[offset]
        output = outputs[cell.name]
        log = logs[cell.name]
        record = _validate_record_identity(
            raw_record,
            cell=cell,
            index=offset + 1,
            output=output,
            log=log,
        )
        status = record.get("status")
        if status == "complete":
            _validate_completed_cell(record, output=output, log=log)
            completed += 1
            continue
        if offset != len(records) - 1 or status not in {"running", "failed"}:
            raise ValueError("only the final non-complete matrix cell can be resumed")
        _archive_interrupted_attempt(progress, record, output=output, log=log)
        break

    progress["cells"] = records[:completed]
    for key in ("status", "failure", "completed_at_utc"):
        progress.pop(key, None)
    progress["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(progress_path, progress)
    return progress, completed


def main() -> None:
    args = _parse_args()
    git = collect_git_metadata(REPO_ROOT, strict=True)
    if git.dirty:
        raise SystemExit("working-set matrix requires a clean Git worktree")
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
    cells = _cells()
    outputs = {cell.name: raw_dir / f"{cell.name}.json" for cell in cells}
    logs = {cell.name: log_dir / f"{cell.name}.log" for cell in cells}
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    git_record = asdict(git)
    plan_record = {
        "path": str(args.plan.resolve()),
        "sha256": sha256_file(args.plan),
    }
    if args.resume:
        if not progress_path.is_file():
            raise SystemExit(f"--resume requires existing matrix progress: {progress_path}")
        progress, completed_cells = _resume_progress(
            progress_path,
            cells=cells,
            outputs=outputs,
            logs=logs,
            git_record=git_record,
            plan_record=plan_record,
        )
    else:
        if progress_path.exists():
            raise SystemExit(f"matrix progress already exists: {progress_path}")
        collisions = [str(path) for path in (*outputs.values(), *logs.values()) if path.exists()]
        if collisions:
            raise SystemExit("refusing to replace matrix artifacts: " + ", ".join(collisions))
        completed_cells = 0
        progress = {
            "record_type": PROGRESS_RECORD_TYPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_record,
            "plan": plan_record,
            "cells": [],
        }
        write_json_atomic(progress_path, progress)

    remaining_paths = [
        path
        for cell in cells[completed_cells:]
        for path in (outputs[cell.name], logs[cell.name])
        if path.exists()
    ]
    if remaining_paths:
        raise SystemExit(
            "refusing to replace remaining matrix artifacts: "
            + ", ".join(str(path) for path in remaining_paths)
        )

    for index, cell in enumerate(cells[completed_cells:], start=completed_cells + 1):
        output = outputs[cell.name]
        log = logs[cell.name]
        command = _command(
            cell,
            model=args.model.resolve(),
            plan=args.plan.resolve(),
            materialized_root=args.materialized_root.resolve(),
            output=output,
            prism_python=_absolute_without_resolving_symlinks(args.prism_python),
            vllm_python=_absolute_without_resolving_symlinks(args.vllm_python),
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
                "output_sha256": sha256_file(output),
            }
        )
        write_json_atomic(progress_path, progress)

    progress["status"] = "complete"
    progress["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(progress_path, progress)
    print(json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
