"""P5.1 visual token importance scoring tests."""

from __future__ import annotations

import json
import subprocess
import sys

from prism_infer.analysis.visual_importance import (
    ImportanceWeights,
    format_importance_markdown,
    score_visual_importance,
)


def _synthetic_trace_records() -> list[dict]:
    return [
        {
            "schema_version": 1,
            "record_type": "trace_header",
            "trace_config": {"metadata": {"case": "p5_1_synthetic"}},
            "model_config": {"model_type": "synthetic"},
        },
        {
            "schema_version": 1,
            "record_type": "attention_layer",
            "step_id": 0,
            "phase": "prefill",
            "layer_id": 0,
            "batch": {
                "input_ids_shape": [6],
                "position_ids_shape": [6],
                "sequences": [
                    {
                        "seq_id": 0,
                        "spans": [
                            {"modality": "text", "start": 0, "end": 2, "index": 0},
                            {"modality": "image", "start": 2, "end": 4, "index": 0},
                            {"modality": "text", "start": 4, "end": 5, "index": 1},
                            {"modality": "video", "start": 5, "end": 6, "index": 0},
                        ],
                    }
                ],
            },
            "span_stats": [
                {
                    "seq_id": 0,
                    "modality": "text",
                    "span_index": 0,
                    "start": 0,
                    "end": 2,
                    "k_norm_mean": 2.0,
                },
                {
                    "seq_id": 0,
                    "modality": "image",
                    "span_index": 0,
                    "start": 2,
                    "end": 4,
                    "k_norm_mean": 4.0,
                },
                {
                    "seq_id": 0,
                    "modality": "video",
                    "span_index": 0,
                    "start": 5,
                    "end": 6,
                    "k_norm_mean": 1.0,
                },
            ],
            "attention": {
                "available": True,
                "kind": "prefill_last_query",
                "sequence_stats": [
                    {
                        "seq_id": 0,
                        "visual_attention_entropy_normalized_mean": 0.25,
                        "span_masses": [
                            {
                                "modality": "text",
                                "span_index": 0,
                                "start": 0,
                                "end": 2,
                                "mass_mean": 0.2,
                            },
                            {
                                "modality": "image",
                                "span_index": 0,
                                "start": 2,
                                "end": 4,
                                "mass_mean": 0.6,
                            },
                            {
                                "modality": "video",
                                "span_index": 0,
                                "start": 5,
                                "end": 6,
                                "mass_mean": 0.2,
                            },
                        ],
                        "top_visual_tokens": [
                            {"token_index": 2, "score": 0.5},
                            {"token_index": 5, "score": 0.2},
                        ],
                    }
                ],
            },
        },
    ]


def test_visual_importance_scores_rank_top_visual_tokens():
    """Attention mass + entropy focus + weak K norm should rank visual tokens."""

    report = score_visual_importance(
        _synthetic_trace_records(),
        weights=ImportanceWeights(attention_mass=1.0, entropy_focus=0.5, k_norm=0.1),
        keep_ratios=(0.5,),
        top_k=3,
    )
    top = report["top_tokens"]
    simulation = report["keep_ratio_simulations"][0]

    print(f"importance source layer records: {report['source']['layer_records']}")
    print(f"importance total visual tokens: {report['total_visual_tokens']}")
    print(f"top token row: {top[0]}")
    print(f"keep ratio simulation: {simulation}")

    assert report["total_visual_tokens"] == 3
    assert report["visual_span_observations"] == 2
    assert top[0]["modality"] == "image"
    assert top[0]["token_index"] == 2
    assert top[0]["score_sum"] > top[1]["score_sum"] > top[2]["score_sum"]
    assert top[0]["top_token_observation_count"] == 1
    assert simulation["keep_count"] == 2
    assert simulation["drop_count"] == 1
    assert simulation["kept_tokens"][0]["token_index"] == 2
    print("P5.1 visual importance ranking: PASS")


def test_visual_importance_includes_image_and_video_spans():
    """Image and video visual spans should both enter the offline ranking."""

    report = score_visual_importance(_synthetic_trace_records(), top_k=10)
    modalities = {row["modality"] for row in report["token_scores"]}
    span_modalities = {row["modality"] for row in report["span_scores"]}
    markdown = format_importance_markdown(report, top_k=2)

    print(f"importance modalities: {sorted(modalities)}")
    print(f"importance span modalities: {sorted(span_modalities)}")
    print(markdown.splitlines()[0])

    assert modalities == {"image", "video"}
    assert span_modalities == {"image", "video"}
    assert "| rank | seq | modality | span | token |" in markdown
    assert "P5.1 Visual Token Importance Report" in markdown
    print("P5.1 visual importance modalities: PASS")


def test_visual_importance_handles_no_visual_tokens():
    """Text-only traces should produce an empty visual ranking, not an error."""

    records = [
        {"schema_version": 1, "record_type": "trace_header"},
        {
            "schema_version": 1,
            "record_type": "attention_layer",
            "step_id": 0,
            "phase": "prefill",
            "layer_id": 0,
            "span_stats": [],
            "attention": {
                "available": True,
                "sequence_stats": [
                    {
                        "seq_id": 0,
                        "visual_attention_entropy_normalized_mean": None,
                        "span_masses": [
                            {
                                "modality": "text",
                                "span_index": 0,
                                "start": 0,
                                "end": 3,
                                "mass_mean": 1.0,
                            }
                        ],
                        "top_visual_tokens": [],
                    }
                ],
            },
        },
    ]

    report = score_visual_importance(records, keep_ratios=(0.25, 1.0), top_k=5)

    print(f"text-only visual tokens: {report['total_visual_tokens']}")
    print(f"text-only keep simulations: {report['keep_ratio_simulations']}")

    assert report["total_visual_tokens"] == 0
    assert report["token_scores"] == []
    assert report["keep_ratio_simulations"][0]["keep_count"] == 0
    assert report["keep_ratio_simulations"][1]["drop_count"] == 0
    print("P5.1 text-only importance empty result: PASS")


def test_score_visual_tokens_cli_writes_json_and_markdown(tmp_path):
    """CLI should read JSONL trace and emit machine/human-readable reports."""

    trace_path = tmp_path / "trace.jsonl"
    output_json = tmp_path / "importance.json"
    output_md = tmp_path / "importance.md"
    with trace_path.open("w", encoding="utf-8") as f:
        for record in _synthetic_trace_records():
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_visual_tokens.py",
            str(trace_path),
            "--output-json",
            str(output_json),
            "--markdown",
            str(output_md),
            "--top-k",
            "2",
            "--keep-ratio",
            "0.5",
        ],
        cwd="/data/Prism-Infer",
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")

    print(f"cli stdout first line: {result.stdout.splitlines()[0]}")
    print(f"cli report visual tokens: {payload['total_visual_tokens']}")
    print(f"cli markdown length: {len(markdown)}")

    assert payload["record_type"] == "visual_importance_report"
    assert payload["total_visual_tokens"] == 3
    assert payload["keep_ratio_simulations"][0]["keep_count"] == 2
    assert "P5.1 Visual Token Importance Report" in result.stdout
    assert "Keep Ratio Simulation" in markdown
    print("P5.1 visual importance CLI: PASS")
