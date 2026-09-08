"""CPU-only report tests, runnable with unittest without importing torch."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import bench_vision_parallel as prism_benchmark
from benchmarks import bench_vision_replicas as replicas_benchmark
from benchmarks import bench_vllm_parallel as benchmark
from benchmarks.vision_parallel_report import save_report, summarize_requests

IMAGE_TOKEN_ID = 151655


def _row(phase: str, start_s: int, finish_s: int) -> dict:
    return {
        "phase": phase,
        "status": "finished",
        "client_start_ns": start_s * 1_000_000_000,
        "client_finish_ns": finish_s * 1_000_000_000,
        "token_ids": [11, 12],
        "observed_token_ids": [11, 12],
        "observed_ids_match_final_ids": True,
        "ttft_ms": 100.0,
        "tpot_ms": 50.0,
        "e2e_ms": (finish_s - start_s) * 1000.0,
    }


class ReportSummaryTests(unittest.TestCase):
    def test_alternating_cold_hot_windows_do_not_publish_sequential_rates(self):
        report = {
            "requests": [
                _row("sequential_cold", 0, 2),
                _row("sequential_hot_new_question", 2, 3),
                _row("sequential_cold", 5, 7),
                _row("sequential_hot_new_question", 7, 8),
            ]
        }
        benchmark._refresh_summaries(report)
        cold = report["phase_summaries"]["sequential_cold"]
        hot = report["phase_summaries"]["sequential_hot_new_question"]
        self.assertEqual(cold["duration_s"], 7.0)
        self.assertEqual(hot["duration_s"], 6.0)
        for summary in (cold, hot):
            self.assertEqual(summary["duration_scope"], "observation_window_including_gaps")
            self.assertNotIn("output_tokens_per_s", summary)
            self.assertNotIn("requests_per_s", summary)
            self.assertEqual(summary["ttft_median_ms"], 100.0)

    def test_concurrent_burst_uses_one_shared_window_and_keeps_raw_rows(self):
        rows = [_row("concurrent_hot", 1, 4), _row("concurrent_hot", 2, 5)]
        original = copy.deepcopy(rows)
        summary = summarize_requests(rows, concurrent=True)
        self.assertEqual(summary["duration_scope"], "single_concurrent_burst")
        self.assertEqual(summary["duration_s"], 4.0)
        self.assertEqual(summary["output_tokens_per_s"], 1.0)
        self.assertEqual(summary["requests_per_s"], 0.5)
        self.assertEqual(rows, original)

    def test_failed_replace_preserves_last_report_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            running = {"status": "running", "requests": [_row("sequential_cold", 0, 1)]}
            save_report(output, running)
            failed = {**running, "status": "failed", "error": {"message": "next batch failed"}}
            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    save_report(output, failed)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), running)
            self.assertEqual(list(Path(directory).iterdir()), [output])
            save_report(output, failed)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), failed)
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_pixel_limit_is_explicit_without_changing_the_old_default(self):
        common = ["--model", "local-model", "--output", "report.json"]
        self.assertEqual(benchmark._parser().parse_args(common).image_max_pixels, 448 * 448)
        larger = benchmark._parser().parse_args(common + ["--image-max-pixels", "802816"])
        self.assertEqual(larger.image_max_pixels, 896 * 896)

    def test_replica_pixel_limit_reaches_the_prism_child(self):
        args = replicas_benchmark._parser().parse_args(
            [
                "--model",
                "local-model",
                "--output-dir",
                "run",
                "--image-max-pixels",
                "802816",
            ]
        )
        command = replicas_benchmark._child_command(args, 0, Path("replica.json"))
        child_args = prism_benchmark._parser().parse_args(command[3:])
        self.assertEqual(child_args.image_max_pixels, 896 * 896)
        self.assertEqual(args.timeout_seconds, 840.0)

    def test_prism_progress_keeps_completed_phases_before_run_finishes(self):
        report = {
            "status": "running",
            "requests": [_row("cold_prefix_disabled", 0, 2)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prism.json"
            prism_benchmark._save_progress(path, report)
            first = json.loads(path.read_text(encoding="utf-8"))
            report["requests"].append(_row("concurrent_hot", 3, 4))
            prism_benchmark._save_progress(path, report)
            latest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(first["status"], "running")
        self.assertEqual(len(first["requests"]), 1)
        self.assertNotIn("output_tokens_per_s", first["phase_summaries"]["cold_prefix_disabled"])
        self.assertEqual(latest["phase_summaries"]["concurrent_hot"]["output_tokens_per_s"], 2.0)


class _FakeEngine:
    def __init__(self, *, fail_after: int | None = None):
        self.started = 0
        self.active = 0
        self.fail_after = fail_after
        self.resets: list[str] = []

    async def reset_prefix_cache(self):
        self.resets.append("prefix")
        return True

    async def reset_encoder_cache(self):
        self.resets.append("encoder")

    async def reset_mm_cache(self):
        self.resets.append("processor")

    async def generate(self, prompt, sampling, request_id):
        if self.started == self.fail_after:
            raise RuntimeError("next batch failed")
        self.started += 1
        self.active += 1
        try:
            for ids, finished in (([11], False), ([12, 13], True)):
                yield SimpleNamespace(
                    prompt_token_ids=[7, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 8],
                    num_cached_tokens=0,
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            token_ids=ids,
                            finish_reason="length" if finished else None,
                        )
                    ],
                    finished=finished,
                )
        finally:
            self.active -= 1


def _arguments(output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        warmup=0,
        repeat=2,
        prompt="cold",
        hot_prompt="hot",
        concurrent_requests=2,
        max_tokens=3,
        replica_id="0",
        wait_for_start=False,
    )


class IncrementalReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_drained_batch_is_saved_outside_request_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            args = _arguments(output)
            engine = _FakeEngine()
            report = {"status": "running", "requests": [], "phase_summaries": {}}
            snapshots = []

            def record_save(path, value):
                self.assertEqual(engine.active, 0)
                self.assertTrue(all(row["status"] == "finished" for row in value["requests"]))
                save_report(path, value)
                snapshots.append(json.loads(output.read_text(encoding="utf-8")))

            with patch.object(benchmark, "save_report", side_effect=record_save):
                with redirect_stdout(io.StringIO()):
                    await benchmark._measure_requests(
                        engine,
                        None,
                        [],
                        {"cold": "cold", "hot": "hot"},
                        args,
                        report,
                        image_token_id=IMAGE_TOKEN_ID,
                    )
            self.assertEqual([len(value["requests"]) for value in snapshots], [1, 2, 3, 4, 6, 8])
            self.assertEqual(engine.resets, ["prefix", "encoder", "processor"] * 3)
            for value in snapshots:
                self.assertEqual(value["status"], "running")
                for row in value["requests"]:
                    self.assertEqual(row["token_ids"], [11, 12, 13])
                    self.assertEqual(row["prompt_image_token_count"], 2)
                    self.assertEqual(row["coalesced_chunk_count"], 1)
            self.assertNotIn(
                "output_tokens_per_s",
                snapshots[-1]["phase_summaries"]["sequential_cold"],
            )
            self.assertIn(
                "output_tokens_per_s",
                snapshots[-1]["phase_summaries"]["concurrent_cold_start"],
            )
            report["status"] = "complete"
            save_report(output, report)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "complete")
            self.assertEqual(list(Path(directory).iterdir()), [output])

    async def test_later_batch_failure_leaves_previous_cold_and_hot_requests_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            report = {"status": "running", "requests": [], "phase_summaries": {}}
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "next batch failed"):
                    await benchmark._measure_requests(
                        _FakeEngine(fail_after=2),
                        None,
                        [],
                        {"cold": "cold", "hot": "hot"},
                        _arguments(output),
                        report,
                        image_token_id=IMAGE_TOKEN_ID,
                    )
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [row["phase"] for row in saved["requests"]],
                ["sequential_cold", "sequential_hot_new_question"],
            )
            self.assertTrue(all(row["status"] == "finished" for row in saved["requests"]))
            report["status"] = "failed"
            report["error"] = {"message": "next batch failed"}
            benchmark._refresh_summaries(report)
            save_report(output, report)
            failed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["requests"][:2], saved["requests"])
            self.assertEqual(len(failed["requests"]), 3)


if __name__ == "__main__":
    unittest.main()
