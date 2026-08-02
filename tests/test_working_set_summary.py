from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from benchmarks.summarize_working_set import (
    UNAVAILABLE,
    build_tables,
    summarize_records,
    write_outputs,
)


def _requests(*, cached_tokens: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "request_id": "request-0",
            "finish_reason": "length",
            "prompt_tokens": 100,
            "output_tokens": 10,
            "ttft_ms": 100.0,
            "latency_ms": 300.0,
        },
        {
            "request_id": "request-1",
            "finish_reason": "stop",
            "prompt_tokens": 100,
            "output_tokens": 10,
            "ttft_ms": 200.0,
            "latency_ms": 500.0,
        },
    ]
    if cached_tokens:
        rows[0]["cached_tokens"] = 50
        rows[1]["cached_tokens"] = 0
    return rows


def _audit(*, variant: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "plan_sha256": "plan-sha",
        "workset_id": "fit",
        "group_ids": ["media-a", "media-b"],
        "dense_prefix_pages": 60,
        "population_requests": 2,
        "measured_requests": 2,
    }
    if variant is not None:
        value["variant"] = variant
    return value


def _population_request(index: int) -> dict[str, object]:
    return {
        "request_id": f"population-{index}",
        "finish_reason": "length",
    }


def _population(engine: str) -> dict[str, object]:
    runs = []
    for index in range(2):
        request = _population_request(index)
        runs.append(
            {"engine_metrics": {"requests": [request]}}
            if engine == "prism"
            else {"requests": [request]}
        )
    population_run = {
        "policy": "one_request_per_group_closed_loop",
        "runs": runs,
    }
    return {"run": population_run} if engine == "prism" else population_run


def _prism_record() -> dict[str, object]:
    return {
        "record_type": "prism_online_run",
        "framework": {"name": "prism-infer"},
        "model": {"config_sha256": "model-sha"},
        "workload": {
            "working_set_plan": _audit(variant="compact_prefix"),
            "prompt_token_ids_sha256": "prompt-fit",
        },
        "arrival": {"trace_sha256": "trace-fit"},
        "engine": {
            "multimodal_prefix_cache": {
                "pre_admission_hits": 2,
                "visual_hydration_skips": 2,
                "stale_probe_fallbacks": 0,
                "hits": 2,
                "misses": 0,
                "evictions": 1,
                "entries": 2,
                "resident_blocks": 36,
                "tail_clone_hits": 1,
                "tail_clone_admissions": 2,
                "tail_clone_evictions": 0,
                "tail_clone_reused_rows": 7,
                "resident_tail_clone_blocks": 2,
            },
            "visual_embedding_cache": {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "entries": 1,
            },
        },
        "visual_compaction": {
            "decisions": 2,
            "effective_reclaims": 2,
            "dense_prompt_blocks": 60,
            "physical_prompt_blocks": 36,
            "released_blocks": 24,
        },
        "memory": {"kv_cache": {"total_bytes": 4_000}},
        "population": _population("prism"),
        "run": {
            "duration_s": 2.0,
            "engine_metrics": {"requests": _requests(cached_tokens=True)},
        },
    }


def _external_record(engine: str) -> dict[str, object]:
    backend = (
        {"kv_cache_memory_bytes_requested": 4_000}
        if engine == "vllm"
        else {"kv_cache_budget_bytes": 4_000}
    )
    return {
        "record_type": "external_online_run",
        "environment": {"framework": engine},
        "model": {"config_sha256": "model-sha"},
        "workload": {
            "working_set_plan": _audit(),
            "prompt_token_ids_sha256": "prompt-fit",
        },
        "arrival": {"trace_sha256": "trace-fit"},
        "backend": backend,
        "population": _population(engine),
        "run": {"duration_s": 2.0, "requests": _requests(cached_tokens=True)},
    }


def _complete_matrix_records() -> list[dict[str, object]]:
    records = []
    for workset_index, workset_id in enumerate(("fit", "knee", "pressure")):
        dense_pages = 60 + workset_index * 20
        group_ids = [f"{workset_id}-media-a", f"{workset_id}-media-b"]
        for variant in ("vision_only", "dense_prefix", "compact_prefix"):
            record = copy.deepcopy(_prism_record())
            audit = record["workload"]["working_set_plan"]
            audit["workset_id"] = workset_id
            audit["variant"] = variant
            audit["dense_prefix_pages"] = dense_pages
            audit["group_ids"] = group_ids
            record["arrival"]["trace_sha256"] = f"trace-{workset_id}"
            record["workload"]["prompt_token_ids_sha256"] = f"prompt-{workset_id}"
            records.append(record)
        for engine in ("vllm", "sglang"):
            record = copy.deepcopy(_external_record(engine))
            audit = record["workload"]["working_set_plan"]
            audit["workset_id"] = workset_id
            audit["dense_prefix_pages"] = dense_pages
            audit["group_ids"] = group_ids
            record["arrival"]["trace_sha256"] = f"trace-{workset_id}"
            record["workload"]["prompt_token_ids_sha256"] = f"prompt-{workset_id}"
            records.append(record)
    return records


class WorkingSetSummaryTest(unittest.TestCase):
    def test_summarizes_measured_metrics_without_inventing_missing_counters(self) -> None:
        summary = summarize_records(
            [_prism_record(), _external_record("vllm"), _external_record("sglang")],
            allow_partial=True,
        )
        cells = {(cell["engine"], cell["variant"]): cell for cell in summary["cells"]}
        prism = cells[("prism", "compact_prefix")]
        vllm = cells[("vllm", "engine_default")]

        self.assertEqual(prism["latency_ms"]["ttft"]["p50"], 150.0)
        self.assertEqual(prism["latency_ms"]["ttft"]["p99"], 199.0)
        self.assertEqual(prism["latency_ms"]["e2e"]["p50"], 400.0)
        self.assertEqual(prism["throughput"]["output_tokens_per_s"], 10.0)
        self.assertEqual(prism["resident_media_entries"], 2)
        self.assertEqual(prism["prefix_cache"]["pre_admission_hits"], 2)
        self.assertEqual(prism["prefix_cache"]["visual_hydration_skips"], 2)
        self.assertEqual(prism["prefix_cache"]["resident_blocks"], 36)
        self.assertEqual(prism["prefix_cache"]["tail_clone_hits"], 1)
        self.assertEqual(prism["prefix_cache"]["evictions"], 1)
        self.assertEqual(prism["compaction"]["actual_compact_pages"], 36)
        self.assertEqual(
            prism["compaction"]["actual_compact_pages_source"],
            "visual_compaction.physical_prompt_blocks",
        )
        self.assertEqual(prism["recomputed_prompt_tokens"], 150)

        self.assertEqual(vllm["prefix_cache"]["hits"], UNAVAILABLE)
        self.assertEqual(vllm["prefix_cache"]["pre_admission_hits"], UNAVAILABLE)
        self.assertEqual(vllm["compaction"]["actual_compact_pages"], UNAVAILABLE)
        self.assertEqual(vllm["vision_cache"]["hits"], UNAVAILABLE)
        self.assertEqual(vllm["cached_token_signal"]["requests_with_cached_tokens"], 1)
        self.assertEqual(vllm["recomputed_prompt_tokens"], 150)

        prism_rows, engine_rows = build_tables(summary)
        self.assertEqual(len(prism_rows), 3)
        self.assertEqual(len(engine_rows), 3)
        missing_dense = next(row for row in prism_rows if row["label"] == "dense_prefix")
        self.assertEqual(missing_dense["ttft_p50_ms"], UNAVAILABLE)

    def test_rejects_incompatible_plan_model_budget_and_trace(self) -> None:
        baseline = _prism_record()
        mutations = {
            "plan SHA256": ("workload", "working_set_plan", "plan_sha256", "other-plan"),
            "model identities": ("model", "config_sha256", None, "other-model"),
            "KV budgets": ("memory", "kv_cache", "total_bytes", 8_000),
            "request traces": ("arrival", "trace_sha256", None, "other-trace"),
            "prompt token": ("workload", "prompt_token_ids_sha256", None, "other-prompt"),
        }
        for message, (first, second, third, value) in mutations.items():
            with self.subTest(message=message):
                changed = copy.deepcopy(baseline)
                target = changed[first][second]
                if third is None:
                    changed[first][second] = value
                else:
                    target[third] = value
                with self.assertRaisesRegex(ValueError, message):
                    summarize_records([baseline, changed])

    def test_requires_complete_matrix_and_complete_population(self) -> None:
        partial = [_prism_record(), _external_record("vllm"), _external_record("sglang")]
        with self.assertRaisesRegex(ValueError, "incomplete 15-cell"):
            summarize_records(partial)

        complete = summarize_records(_complete_matrix_records())
        self.assertEqual(complete["matrix"]["required_cells"], 15)
        self.assertEqual(complete["matrix"]["observed_cells"], 15)
        self.assertTrue(complete["matrix"]["complete"])

        incomplete_population = _prism_record()
        incomplete_population["population"]["run"]["runs"][1]["engine_metrics"]["requests"][0][
            "finish_reason"
        ] = None
        with self.assertRaisesRegex(ValueError, "population request.*did not complete"):
            summarize_records([incomplete_population], allow_partial=True)

    def test_writes_the_requested_artifacts(self) -> None:
        summary = summarize_records(
            [_prism_record(), _external_record("vllm"), _external_record("sglang")],
            allow_partial=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_outputs(summary, directory)

            self.assertEqual(
                set(outputs),
                {
                    "summary_json",
                    "main_png",
                    "prism_csv",
                    "prism_markdown",
                    "engines_csv",
                    "engines_markdown",
                },
            )
            self.assertEqual(Path(outputs["main_png"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIn(
                "unavailable",
                Path(outputs["engines_csv"]).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
