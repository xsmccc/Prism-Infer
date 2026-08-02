"""Focused coverage for resumable working-set matrix execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.run_working_set_matrix import (
    PROGRESS_RECORD_TYPE,
    MatrixCell,
    _resume_progress,
)


def test_resume_preserves_completed_cell_and_archives_interrupted_log(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    log_dir = tmp_path / "logs"
    raw_dir.mkdir()
    log_dir.mkdir()
    cells = (
        MatrixCell("prism", "fit", "vision_only"),
        MatrixCell("prism", "fit", "dense_prefix"),
    )
    outputs = {cell.name: raw_dir / f"{cell.name}.json" for cell in cells}
    logs = {cell.name: log_dir / f"{cell.name}.log" for cell in cells}

    completed_output = outputs[cells[0].name]
    completed_output.write_text('{"complete": true}\n', encoding="utf-8")
    logs[cells[0].name].write_text("complete\n", encoding="utf-8")
    logs[cells[1].name].write_text("interrupted\n", encoding="utf-8")
    git_record = {"commit": "abc", "dirty": False}
    plan_record = {"path": "/plan.json", "sha256": "def"}
    records = [
        {
            "index": 1,
            "engine": cells[0].engine,
            "workset": cells[0].workset,
            "variant": cells[0].variant,
            "name": cells[0].name,
            "output": str(completed_output),
            "log": str(logs[cells[0].name]),
            "status": "complete",
            "output_sha256": hashlib.sha256(completed_output.read_bytes()).hexdigest(),
        },
        {
            "index": 2,
            "engine": cells[1].engine,
            "workset": cells[1].workset,
            "variant": cells[1].variant,
            "name": cells[1].name,
            "output": str(outputs[cells[1].name]),
            "log": str(logs[cells[1].name]),
            "status": "running",
        },
    ]
    progress_path = tmp_path / "matrix_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "record_type": PROGRESS_RECORD_TYPE,
                "git": git_record,
                "plan": plan_record,
                "cells": records,
            }
        ),
        encoding="utf-8",
    )

    progress, completed = _resume_progress(
        progress_path,
        cells=cells,
        outputs=outputs,
        logs=logs,
        git_record=git_record,
        plan_record=plan_record,
    )

    assert completed == 1
    assert progress["cells"] == records[:1]
    assert completed_output.is_file()
    assert logs[cells[0].name].is_file()
    assert not logs[cells[1].name].exists()
    archived = log_dir / f"{cells[1].name}.interrupted-1.log"
    assert archived.read_text(encoding="utf-8") == "interrupted\n"
    assert progress["interrupted_attempts"][0]["archived_log"] == str(archived)
