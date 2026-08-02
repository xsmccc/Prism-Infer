from __future__ import annotations

import copy
import unittest

from benchmarks.trace_working_set_prefix import (
    compact_prism_config,
    select_knee_trace_contract,
    validate_prefix_hit_evidence,
)


def _plan() -> dict[str, object]:
    return {
        "model": {"max_model_len": 8192},
        "kv_budget": {"pages": 220, "page_size_tokens": 256},
        "processor": {"image_max_pixels": 602_112},
        "serving": {"max_num_seqs": 8, "max_chunk_size": 8192},
        "groups": [
            {
                "group_id": "warmup-group",
                "ordered_media_sha256": ["warmup-media"],
                "dense_prefix_pages": 20,
                "samples": [
                    {
                        "sample_id": "warmup-question",
                        "source_prompt": "Warm up this other image.",
                    }
                ],
            },
            {
                "group_id": "target-group",
                "ordered_media_sha256": ["target-media"],
                "dense_prefix_pages": 40,
                "samples": [
                    {"sample_id": "target-cold", "source_prompt": "First question?"},
                    {"sample_id": "target-hit", "source_prompt": "Second question?"},
                ],
            },
        ],
        "worksets": [
            {"id": "fit", "group_ids": ["warmup-group"]},
            {
                "id": "knee",
                "group_ids": ["warmup-group", "target-group"],
                "measured_requests": [{"group_id": "target-group", "sample_id": "target-hit"}],
            },
            {"id": "pressure", "group_ids": ["warmup-group", "target-group"]},
        ],
    }


def _metadata() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    before = {
        "pre_admission_hits": 0,
        "visual_hydration_skips": 0,
        "stale_probe_fallbacks": 0,
        "hits": 0,
        "misses": 0,
        "admissions": 0,
        "evictions": 0,
    }
    after_cold = {
        **before,
        "misses": 1,
        "admissions": 1,
    }
    after_hit = {
        **after_cold,
        "pre_admission_hits": 1,
        "visual_hydration_skips": 1,
        "hits": 1,
    }
    return before, after_cold, after_hit


class WorkingSetPrefixTraceTest(unittest.TestCase):
    def test_selects_multi_question_target_and_distinct_warmup(self) -> None:
        selection = select_knee_trace_contract(_plan())
        config = compact_prism_config(_plan())

        self.assertEqual(selection["workset_id"], "knee")
        self.assertEqual(selection["target"]["group_id"], "target-group")
        self.assertEqual(selection["target"]["cold"]["sample_id"], "target-cold")
        self.assertEqual(selection["target"]["prefix_hit"]["sample_id"], "target-hit")
        self.assertEqual(selection["warmup"]["group_id"], "warmup-group")
        self.assertEqual(config["mode"], "visual_compact_scaled_fp8_compile_graph")
        self.assertEqual(config["tensor_parallel_size"], 1)
        self.assertEqual(config["num_kvcache_blocks"], 220)
        self.assertEqual(config["max_num_seqs"], 8)
        self.assertEqual(config["visual_pruning_keep_ratio"], 0.6)

    def test_requires_observed_non_stale_hit_and_cached_tokens(self) -> None:
        before, after_cold, after_hit = _metadata()
        evidence = validate_prefix_hit_evidence(
            before,
            after_cold,
            after_hit,
            {
                "finish_reason": "length",
                "cached_tokens": 0,
                "prompt_tokens": 1024,
            },
            {
                "finish_reason": "length",
                "cached_tokens": 768,
                "prompt_tokens": 1024,
            },
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["actual_cached_tokens"], 768)
        self.assertEqual(
            evidence["prefix_hit_request_counter_deltas"]["visual_hydration_skips"],
            1,
        )
        self.assertFalse(evidence["stale_probe_fallback"])

    def test_trace_pair_must_exist_in_measured_schedule(self) -> None:
        plan = _plan()
        plan["worksets"][1]["measured_requests"] = [
            {"group_id": "warmup-group", "sample_id": "warmup-question"}
        ]

        with self.assertRaisesRegex(ValueError, "no media group with two distinct questions"):
            select_knee_trace_contract(plan)

    def test_rejects_stale_fallback_or_missing_cached_tokens(self) -> None:
        before, after_cold, after_hit = _metadata()
        stale = copy.deepcopy(after_hit)
        stale["stale_probe_fallbacks"] = 1
        with self.assertRaisesRegex(ValueError, "stale prefix probe"):
            validate_prefix_hit_evidence(
                before,
                after_cold,
                stale,
                {"finish_reason": "length", "cached_tokens": 0, "prompt_tokens": 1024},
                {"finish_reason": "length", "cached_tokens": 768, "prompt_tokens": 1024},
            )
        with self.assertRaisesRegex(ValueError, "no actual cached tokens"):
            validate_prefix_hit_evidence(
                before,
                after_cold,
                after_hit,
                {"finish_reason": "length", "cached_tokens": 0, "prompt_tokens": 1024},
                {"finish_reason": "length", "cached_tokens": 0, "prompt_tokens": 1024},
            )


if __name__ == "__main__":
    unittest.main()
