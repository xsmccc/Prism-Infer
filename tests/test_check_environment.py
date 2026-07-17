import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("prism_check_environment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
inspect_model = MODULE.inspect_model


def test_inspect_model_not_requested() -> None:
    result, errors = inspect_model(None)

    assert result == {"status": "NOT_CHECKED", "path": None}
    assert errors == []


def test_inspect_model_accepts_minimal_qwen3_vl_snapshot(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    result, errors = inspect_model(str(tmp_path))

    assert errors == []
    assert result["status"] == "PASS"
    assert result["model_type"] == "qwen3_vl"
    assert result["weight_files"] == 1
    assert len(result["config_sha256"]) == 64


def test_inspect_model_reports_missing_files_and_wrong_type(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2Model"]}),
        encoding="utf-8",
    )

    result, errors = inspect_model(str(tmp_path))

    assert result["status"] == "FAIL"
    assert result["missing_files"] == [
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    assert any("no *.safetensors" in error for error in errors)
    assert any("not identified as Qwen3-VL" in error for error in errors)
