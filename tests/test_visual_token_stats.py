"""P4 KV trace 离线统计验证。"""

from prism_infer.analysis.kv_trace import (
    format_summary_markdown,
    render_summary_svg,
    summarize_trace,
)


def _record(layer_id: int, visual_mass: float, visual_k: float, text_k: float) -> dict:
    return {
        "schema_version": 1,
        "record_type": "attention_layer",
        "step_id": 0,
        "phase": "prefill",
        "layer_id": layer_id,
        "head_stats": {
            "by_modality": {
                "image": {
                    "token_count": 2,
                    "k_norm_mean": visual_k,
                    "v_norm_mean": visual_k + 0.5,
                    "k_norm_mean_by_head": [visual_k, visual_k + 1.0],
                    "v_norm_mean_by_head": [visual_k + 0.5, visual_k + 1.5],
                },
                "text": {
                    "token_count": 3,
                    "k_norm_mean": text_k,
                    "v_norm_mean": text_k + 0.5,
                    "k_norm_mean_by_head": [text_k, text_k + 1.0],
                    "v_norm_mean_by_head": [text_k + 0.5, text_k + 1.5],
                },
            }
        },
        "attention": {
            "available": True,
            "kind": "prefill_last_query",
            "sequence_stats": [
                {
                    "seq_id": 0,
                    "visual_mass_mean": visual_mass,
                    "text_mass_mean": 1.0 - visual_mass,
                    "attention_entropy_mean": 1.25 + layer_id,
                    "attention_entropy_normalized_mean": 0.5 + layer_id * 0.1,
                    "visual_attention_entropy_normalized_mean": 0.4 + layer_id * 0.1,
                    "head_visual_mass": [visual_mass - 0.1, visual_mass + 0.1],
                    "head_text_mass": [1.1 - visual_mass, 0.9 - visual_mass],
                }
            ],
        },
    }


def test_summarize_trace_reports_attention_norm_ratio_and_redundancy():
    """summary 应输出 visual attention、KV norm ratio 和层间相似性。"""

    records = [
        {
            "schema_version": 1,
            "record_type": "trace_header",
            "trace_config": {"metadata": {"case": "synthetic"}},
            "model_config": {"model_type": "synthetic"},
        },
        _record(0, 0.25, 4.0, 2.0),
        _record(1, 0.50, 6.0, 3.0),
    ]
    summary = summarize_trace(records)
    markdown = format_summary_markdown(summary)
    svg = render_summary_svg(summary)

    print(f"summary layers: {summary['layers']}")
    print(f"summary phases: {summary['phases']}")
    print(f"layer0 ratio: {summary['per_layer']['0']['visual_text_k_norm_ratio']:.6e}")
    print(f"layer1 visual mass: {summary['per_layer']['1']['visual_attention_mass_mean']:.6e}")
    print(markdown)
    print(svg.splitlines()[0])

    assert summary["num_layer_records"] == 2
    assert summary["phases"] == ["prefill"]
    assert summary["per_layer"]["0"]["visual_text_k_norm_ratio"] == 2.0
    assert summary["per_layer"]["1"]["visual_attention_mass_mean"] == 0.5
    assert summary["per_layer"]["0"]["attention_entropy_mean"] == 1.25
    assert summary["per_layer"]["1"]["attention_entropy_normalized_mean"] == 0.6
    assert summary["adjacent_layer_redundancy"][0]["visual_k_head_cosine"] > 0.99
    assert "| layer | visual attn mass | text attn mass | entropy |" in markdown
    assert svg.startswith("<svg ")
    assert "KV Trace Layer Summary" in svg
    print("KV trace visual token summary: PASS")
