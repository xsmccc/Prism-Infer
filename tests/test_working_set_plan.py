from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from benchmarks.working_set_workload import (
    materialize_working_set,
    source_prompt_schedule_sha256,
    verify_working_set_model,
    verify_working_set_processor,
    working_set_processor_kwargs,
)
from prism_infer.analysis.working_set_plan import (
    DEFAULT_IMAGE_MAX_PIXELS,
    DEFAULT_IMAGE_MIN_PIXELS,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MAX_NUM_SEQS,
    DENSE_PREFIX_PAGES_RECORD_TYPE,
    build_media_first_groups,
    build_working_set_plan,
    load_dense_prefix_pages,
    load_muirbench_records,
    load_working_set_plan,
    validate_working_set_plan,
    write_working_set_plan,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(*, groups: int = 6, questions_per_group: int = 2) -> list[dict[str, object]]:
    records = []
    for group_index in range(groups):
        media_digest = _digest(f"media-{group_index}")
        for question_index in range(questions_per_group):
            records.append(
                {
                    "sample_id": f"sample-{group_index}-{question_index}",
                    "question": (
                        f"<image> Compare the content for group {group_index}, "
                        f"question {question_index}."
                    ),
                    "options": ["first", "second", "third", "fourth"],
                    "media": [
                        {
                            "sha256": media_digest,
                            "materialized_path": f"media/{media_digest}.png",
                            "width": 64,
                            "height": 64,
                        }
                    ],
                }
            )
    return records


def _page_map(records: list[dict[str, object]], pages: int = 30) -> dict[str, int]:
    return {group["group_id"]: pages for group in build_media_first_groups(records)}


class WorkingSetPlanTest(unittest.TestCase):
    def test_groups_exact_ordered_media_and_builds_media_first_prompts(self) -> None:
        records = _records(groups=3)
        records[0]["question"] = "Compare the content and choose."
        records[0]["options"][0] = "<image> first"
        groups = build_media_first_groups(records)

        self.assertEqual(
            [group["group_id"] for group in groups], sorted(group["group_id"] for group in groups)
        )
        self.assertEqual([len(group["samples"]) for group in groups], [2, 2, 2])
        for group in groups:
            self.assertEqual(
                group["group_id"],
                _digest(json.dumps(group["ordered_media_sha256"], separators=(",", ":"))),
            )
            self.assertEqual(group["ordered_media_sha256"][0], group["media"][0]["sha256"])
            self.assertNotIn("<image>", group["samples"][0]["source_prompt"])
            self.assertIn("Image 1", group["samples"][0]["source_prompt"])
            self.assertEqual(group["samples"][0]["sample_offset"], 0)
            self.assertEqual(group["samples"][1]["sample_offset"], 1)

    def test_plan_selects_boundaries_and_generates_deterministic_traffic(self) -> None:
        records = _records(groups=6)
        kwargs = {
            "dense_prefix_pages": _page_map(records),
            "model_revision": "revision-1",
            "kv_budget_bytes": 1_000_000,
            "kv_budget_pages": 100,
        }
        plan = build_working_set_plan(records, **kwargs)
        repeated = build_working_set_plan(records, **kwargs)

        self.assertEqual(plan, repeated)
        self.assertEqual(plan["model"]["max_model_len"], 8192)
        self.assertEqual(plan["traffic"]["arrival_process"], "poisson")
        self.assertEqual(plan["traffic"]["request_rate_per_s"], 4.0)
        self.assertEqual(plan["traffic"]["zipf_alpha"], 1.0)
        self.assertEqual(plan["traffic"]["seed"], 20260801)
        self.assertEqual(plan["traffic"]["measured_requests"], 600)
        self.assertEqual(plan["traffic"]["max_new_tokens"], 16)
        self.assertEqual(plan["processor"]["image_min_pixels"], DEFAULT_IMAGE_MIN_PIXELS)
        self.assertEqual(plan["processor"]["image_max_pixels"], DEFAULT_IMAGE_MAX_PIXELS)
        self.assertEqual(plan["serving"]["max_num_seqs"], DEFAULT_MAX_NUM_SEQS)
        self.assertEqual(plan["serving"]["max_chunk_size"], DEFAULT_MAX_CHUNK_SIZE)
        self.assertNotEqual(
            plan["traffic"]["rng_streams"]["group_selection_seed"],
            plan["traffic"]["rng_streams"]["arrival_seed"],
        )
        worksets = {workset["id"]: workset for workset in plan["worksets"]}
        self.assertEqual(worksets["fit"]["dense_prefix_pages"], 60)
        self.assertEqual(worksets["knee"]["dense_prefix_pages"], 120)
        self.assertEqual(worksets["pressure"]["dense_prefix_pages"], 150)
        self.assertEqual(worksets["fit"]["groups"], 2)
        self.assertEqual(worksets["knee"]["groups"], 4)
        self.assertEqual(worksets["pressure"]["groups"], 5)
        for workset in worksets.values():
            self.assertEqual(len(workset["population_requests"]), workset["groups"])
            self.assertEqual(len(workset["measured_requests"]), 600)
            offsets = [request["arrival_offset_s"] for request in workset["measured_requests"]]
            self.assertEqual(offsets, sorted(offsets))
            self.assertGreater(offsets[0], 0.0)
            first_by_group = {}
            for request in workset["measured_requests"]:
                first_by_group.setdefault(request["group_id"], request)
            self.assertTrue(
                all(request["sample_offset"] == 1 for request in first_by_group.values())
            )

    def test_pressure_uses_all_groups_when_150_percent_is_unavailable(self) -> None:
        records = _records(groups=4)
        groups = build_media_first_groups(records)
        pages = {
            group["group_id"]: value for group, value in zip(groups, (30, 30, 30, 40), strict=True)
        }
        plan = build_working_set_plan(
            records,
            dense_prefix_pages=pages,
            model_revision="revision-1",
            kv_budget_pages=100,
            measured_requests=8,
        )
        pressure = next(workset for workset in plan["worksets"] if workset["id"] == "pressure")

        self.assertEqual(pressure["groups"], 4)
        self.assertEqual(pressure["dense_prefix_pages"], 130)
        self.assertFalse(pressure["target_reached"])

    def test_loaders_bind_materialization_selection_and_page_artifact(self) -> None:
        records = _records(groups=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_path = root / "records/muirbench_test.final.jsonl"
            records_path.parent.mkdir(parents=True)
            records_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            sample_ids = [record["sample_id"] for record in records]
            artifact = {
                "id": "muirbench_test",
                "selected_records": {
                    "path": "records/muirbench_test.final.jsonl",
                    "sha256": _digest_file(records_path),
                },
                "selection": {"final": {"sample_ids": sample_ids}},
            }
            manifest_path = root / "p9_quality_materialization.json"
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "datasets": [artifact]}),
                encoding="utf-8",
            )
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "id": "muirbench_test",
                                "selection": {"final": {"sample_ids": sample_ids}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loaded, identity = load_muirbench_records(
                root,
                selection_path=selection_path,
            )
            groups = build_media_first_groups(loaded)
            page_artifact = root / "dense_pages.json"
            page_artifact.write_text(
                json.dumps(
                    {
                        "record_type": DENSE_PREFIX_PAGES_RECORD_TYPE,
                        "groups": [
                            {"group_id": group["group_id"], "dense_prefix_pages": 50}
                            for group in groups
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pages = load_dense_prefix_pages(page_artifact)
            plan = build_working_set_plan(
                loaded,
                dense_prefix_pages=pages,
                model_revision="revision-1",
                materialization_identity=identity,
                kv_budget_pages=100,
                measured_requests=8,
            )
            plan_path = root / "plan.json"
            output_sha256 = write_working_set_plan(plan_path, plan)

            self.assertEqual(loaded, records)
            self.assertEqual(identity["selected_records_sha256"], _digest_file(records_path))
            self.assertEqual(output_sha256, _digest_file(plan_path))
            self.assertEqual(load_working_set_plan(plan_path), plan)

    def test_validator_rejects_changed_request_schedule(self) -> None:
        records = _records(groups=6)
        plan = build_working_set_plan(
            records,
            dense_prefix_pages=_page_map(records),
            model_revision="revision-1",
            kv_budget_pages=100,
            measured_requests=8,
        )
        changed = copy.deepcopy(plan)
        changed["worksets"][0]["measured_requests"][0]["sample_offset"] = 0

        with self.assertRaisesRegex(ValueError, "deterministic contract"):
            validate_working_set_plan(changed)

    def test_shared_materializer_uses_one_images_shape_for_all_engines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for group_index in range(6):
                media_path = root / f"media/{group_index}.png"
                media_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (2, 2), color=(group_index, 0, 0)).save(media_path)
                media_digest = _digest_file(media_path)
                for question_index in range(2):
                    records.append(
                        {
                            "sample_id": f"sample-{group_index}-{question_index}",
                            "question": f"<image> question {question_index}",
                            "options": ["first", "second"],
                            "media": [
                                {
                                    "sha256": media_digest,
                                    "materialized_path": f"media/{group_index}.png",
                                }
                            ],
                        }
                    )
            plan = build_working_set_plan(
                records,
                dense_prefix_pages=_page_map(records),
                model_revision="revision-1",
                kv_budget_pages=100,
                measured_requests=8,
            )
            plan_path = root / "plan.json"
            write_working_set_plan(plan_path, plan)

            materialized = materialize_working_set(
                plan_path,
                workset_id="fit",
                materialized_root=root,
            )
            try:
                payloads = materialized.population_payloads + materialized.measured_payloads
                self.assertTrue(payloads)
                self.assertTrue(all(payload["type"] == "images" for payload in payloads))
                self.assertTrue(all(len(payload["images"]) == 1 for payload in payloads))
                self.assertTrue(all("image" not in payload for payload in payloads))
                self.assertEqual(
                    materialized.population_group_ids,
                    [
                        request["group_id"]
                        for request in materialized.workset["population_requests"]
                    ],
                )
                self.assertEqual(
                    materialized.measured_group_ids,
                    [request["group_id"] for request in materialized.workset["measured_requests"]],
                )
                prompt_digest = source_prompt_schedule_sha256(materialized.measured_payloads)
                changed_payloads = copy.deepcopy(materialized.measured_payloads)
                changed_payloads[0]["prompt"] += " changed"
                self.assertNotEqual(
                    prompt_digest,
                    source_prompt_schedule_sha256(changed_payloads),
                )
            finally:
                materialized.close()

            model_path = root / "revision-1"
            model_path.mkdir()
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            model_verification = verify_working_set_model(plan, model_path)
            self.assertEqual(model_verification["expected_revision"], "revision-1")
            self.assertEqual(model_verification["matched_by"], "path_component")

            processor_kwargs = working_set_processor_kwargs(plan)
            processor = SimpleNamespace(
                image_processor=SimpleNamespace(
                    size=SimpleNamespace(
                        shortest_edge=processor_kwargs["min_pixels"],
                        longest_edge=processor_kwargs["max_pixels"],
                    )
                )
            )
            self.assertEqual(
                verify_working_set_processor(processor, plan)["image_max_pixels"],
                DEFAULT_IMAGE_MAX_PIXELS,
            )

            wrong_model_path = root / "different-revision"
            wrong_model_path.mkdir()
            (wrong_model_path / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                verify_working_set_model(plan, wrong_model_path)


if __name__ == "__main__":
    unittest.main()
