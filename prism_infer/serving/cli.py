"""Command-line entry point for the Prism-Infer native HTTP server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_infer import LLM
from prism_infer.serving.app import create_app
from prism_infer.serving.runtime import ServingEngine, ServingRuntime


def _load_engine_options(path: str | None) -> dict[str, object]:
    """读取直接传给 ``LLM`` 的 JSON 配置对象。"""

    if path is None:
        return {}
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read engine config {config_path}: {exc}") from exc
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("engine config must be a JSON object with string keys")
    protected = sorted({"model", "metrics_sink", "clock_ns", "request_id_allocator"} & payload.keys())
    if protected:
        raise ValueError(f"engine config cannot override CLI-owned fields: {protected}")
    return payload


def _engine_factory(
    model: str,
    engine_options: dict[str, object],
) -> ServingEngine:
    """在专用所有者线程内构造实际 Prism-Infer 引擎。"""

    return LLM(model, **engine_options)


def _parser() -> argparse.ArgumentParser:
    """构造稳定且刻意精简的服务启动参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local Qwen3-VL model directory")
    parser.add_argument(
        "--engine-config",
        help="JSON object containing Config keyword arguments",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ingress-capacity", type=int, default=64)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def main() -> None:
    """启动单 worker ASGI server；GPU 引擎由内部所有者线程独占。"""

    parser = _parser()
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.ingress_capacity <= 0:
        parser.error("--ingress-capacity must be positive")
    try:
        engine_options = _load_engine_options(args.engine_config)
    except ValueError as exc:
        parser.error(str(exc))

    runtime = ServingRuntime(
        lambda: _engine_factory(args.model, engine_options),
        ingress_capacity=args.ingress_capacity,
    )
    app = create_app(runtime)

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "network serving requires `pip install 'prism-infer[serving]'`"
        ) from exc
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
