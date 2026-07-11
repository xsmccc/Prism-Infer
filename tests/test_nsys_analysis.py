"""P6.2 Nsight Systems SQLite structured analyzer 测试。"""

import sqlite3
from pathlib import Path

from benchmarks.analyze_nsys import analyze_nsys_sqlite


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
            start INTEGER, end INTEGER, correlationId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, correlationId INTEGER, nameId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
            start INTEGER, end INTEGER, correlationId INTEGER
        );
        INSERT INTO StringIds VALUES (1, 'prism::engine.model_runner');
        INSERT INTO StringIds VALUES (2, 'cudaLaunchKernel_v7000');
        INSERT INTO StringIds VALUES (3, 'cudaMemcpyAsync_v3020');
        INSERT INTO StringIds VALUES (4, 'cudaStreamSynchronize_v3020');
        INSERT INTO StringIds VALUES (5, 'cudaGraphLaunch_v10000');
        INSERT INTO StringIds VALUES (6, 'prism::target');
        INSERT INTO NVTX_EVENTS VALUES (0, 1000, NULL, 1);
        INSERT INTO NVTX_EVENTS VALUES (2000, 3000, NULL, 1);
        INSERT INTO NVTX_EVENTS VALUES (4000, 5000, NULL, 1);
        INSERT INTO NVTX_EVENTS VALUES (2100, 2500, NULL, 6);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (200, 300, 10);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2200, 2300, 20);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (4200, 4400, 30);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 150, 10, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2100, 2150, 20, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (4100, 4150, 30, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2160, 2170, 40, 3);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2180, 2190, 41, 4);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (4050, 4060, 42, 5);
        INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (4250, 4350, 42);
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
    assert result["phase_summary"]["decode"]["kernel_count"]["median"] == 1
    assert result["phase_summary"]["decode"]["memcpy_async_count"]["max"] == 1
    assert result["phase_summary"]["decode"]["stream_synchronize_count"]["max"] == 1
    assert result["phase_summary"]["decode"]["graph_launch_api_count"]["max"] == 1
    assert result["phase_summary"]["decode"]["graph_execution_ms"]["max"] == 0.0001
    assert result["target_ranges"]["prism::target"]["kernel_count_total"] == 1
    assert result["target_ranges"]["prism::target"]["memcpy_async_count_total"] == 1
    assert (
        result["target_ranges"]["prism::target"][
            "stream_synchronize_count_total"
        ]
        == 1
    )
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
