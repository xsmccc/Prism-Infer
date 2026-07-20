"""P6.2 Nsight Systems SQLite structured analyzer 测试。"""

import sqlite3
import sys
from pathlib import Path

import pytest

from benchmarks.analyze_nsys import analyze_nsys_sqlite, main


def _build_synthetic_nsys_sqlite(path: Path) -> None:
    """建立包含一个 prefill 和两个 decode range 的最小 Nsight fixture。"""

    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, text TEXT, textId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, correlationId INTEGER,
            demangledName INTEGER, shortName INTEGER, mangledName INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, correlationId INTEGER, nameId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
            start INTEGER, end INTEGER, correlationId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, correlationId INTEGER, bytes INTEGER
        );
        INSERT INTO StringIds VALUES (1, 'prism::engine.model_runner');
        INSERT INTO StringIds VALUES (2, 'cudaLaunchKernel_v7000');
        INSERT INTO StringIds VALUES (3, 'cudaMemcpyAsync_v3020');
        INSERT INTO StringIds VALUES (4, 'cudaStreamSynchronize_v3020');
        INSERT INTO StringIds VALUES (5, 'cudaGraphLaunch_v10000');
        INSERT INTO StringIds VALUES (6, 'prism::target');
        INSERT INTO StringIds VALUES (7, 'internal::gemvx::kernel');
        INSERT INTO StringIds VALUES (8, '_paged_decode_attention_kernel');
        INSERT INTO NVTX_EVENTS VALUES (0, 1000, NULL, 1);
        INSERT INTO NVTX_EVENTS VALUES (2000, 3000, NULL, 1);
        INSERT INTO NVTX_EVENTS VALUES (4000, 5000, NULL, 1);
        INSERT INTO NVTX_EVENTS VALUES (2100, 2250, NULL, 6);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (200, 300, 10, 8, NULL, NULL);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2200, 2300, 20, 7, NULL, NULL);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (4200, 4400, 30, 8, NULL, NULL);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 150, 10, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2100, 2150, 20, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (4100, 4150, 30, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2160, 2170, 40, 3);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2180, 2190, 41, 4);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (4050, 4060, 42, 5);
        INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (4250, 4350, 42);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (2210, 2220, 40, 4096);
        """
    )
    connection.commit()
    connection.close()


def test_analyze_nsys_sqlite_reports_phase_and_target_metrics(tmp_path: Path) -> None:
    """analyzer 应按 phase 聚合 API/kernel，并解析 target NVTX range。"""

    sqlite_path = tmp_path / "synthetic.sqlite"
    _build_synthetic_nsys_sqlite(sqlite_path)

    result = analyze_nsys_sqlite(
        sqlite_path,
        target_ranges=["prism::target"],
    )

    assert len(result["raw_steps"]) == 3
    assert result["raw_steps"][0]["phase"] == "prefill"
    assert result["phase_summary"]["prefill"]["cpu_range_ms"]["max"] == 0.001
    assert result["phase_summary"]["decode"]["kernel_count"]["median"] == 1
    assert result["phase_summary"]["decode"]["memcpy_async_count"]["max"] == 1
    assert result["phase_summary"]["decode"]["stream_synchronize_count"]["max"] == 1
    assert result["phase_summary"]["decode"]["graph_launch_api_count"]["max"] == 1
    assert result["phase_summary"]["decode"]["graph_execution_ms"]["max"] == 0.0001
    assert result["target_ranges"]["prism::target"]["kernel_count_total"] == 1
    assert result["target_ranges"]["prism::target"]["memcpy_async_count_total"] == 1
    assert result["schema_version"] == 2
    target = result["target_ranges"]["prism::target"]
    assert target["memcpy_bytes_total"] == 4096
    assert target["gpu_busy_ms_per_range"]["median"] == 0.0001
    assert target["gpu_span_ms_per_range"]["median"] == 0.0001
    assert target["cpu_gpu_busy_overlap_ms_per_range"]["median"] == 0.00005
    assert target["gpu_tail_after_cpu_ms_per_range"]["median"] == 0.00005
    assert target["kernel_categories"]["linear_gemv"]["kernel_count_total"] == 1
    assert target["kernel_categories"]["linear_gemv"]["kernel_time_fraction"] == 1
    assert target["top_kernels"][0]["name"] == "internal::gemvx::kernel"
    assert result["target_ranges"]["prism::target"]["stream_synchronize_count_total"] == 1
    connection = sqlite3.connect(sqlite_path)
    persisted_analysis_indexes = connection.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'analysis_%'"
    ).fetchall()
    connection.close()
    assert persisted_analysis_indexes == []
    print("P6.2 Nsight SQLite phase/target analysis: PASS")


def test_analyze_nsys_sqlite_accepts_capture_without_graph_table(
    tmp_path: Path,
) -> None:
    """eager capture 没有 graph trace 表时应明确记录 0，而不是拒绝文件。"""

    sqlite_path = tmp_path / "eager.sqlite"
    _build_synthetic_nsys_sqlite(sqlite_path)
    connection = sqlite3.connect(sqlite_path)
    connection.execute("DROP TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
    connection.commit()
    connection.close()

    result = analyze_nsys_sqlite(sqlite_path)

    assert result["phase_summary"]["prefill"]["graph_execution_ms"]["max"] == 0
    assert result["phase_summary"]["decode"]["graph_execution_ms"]["max"] == 0
    print("P6.2 Nsight eager capture without graph table: PASS")


def test_analyzer_cli_quietly_writes_once_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """summary artifact 默认不得覆盖，quiet 模式也必须保留机器可读输出。"""

    sqlite_path = tmp_path / "synthetic.sqlite"
    output_path = tmp_path / "summary.json"
    _build_synthetic_nsys_sqlite(sqlite_path)
    argv = [
        "analyze_nsys.py",
        str(sqlite_path),
        "--output",
        str(output_path),
        "--quiet",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    assert output_path.is_file()
    assert capsys.readouterr().out == ""
    original = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()
    assert output_path.read_bytes() == original
