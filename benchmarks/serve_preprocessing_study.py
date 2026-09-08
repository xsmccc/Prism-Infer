"""Start the normal HTTP server and save its cache counters on shutdown.

PYTHONPATH selects the source checkout; this runner does not prepend its own
repository. --nvtx only enables existing semantic ranges for a separate trace run.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import torch
import uvicorn

import prism_infer
from prism_infer import LLM
from prism_infer.observability.performance import install_performance_provider
from prism_infer.serving.app import create_app
from prism_infer.serving.runtime import ServingRuntime


@contextmanager
def nvtx_region(name, *, cuda=True, metadata=None):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    options = json.loads(args.config.read_text(encoding="utf-8"))
    if args.pid_file is not None:
        args.pid_file.write_text(str(os.getpid()), encoding="utf-8")
    if args.nvtx:
        install_performance_provider(
            profile_region_provider=nvtx_region,
            profile_session_provider=lambda: None,
        )
    engines = []

    def factory():
        engine = LLM(args.model, **options)
        engines.append(engine)
        return engine

    runtime = ServingRuntime(factory, ingress_capacity=16)
    app = create_app(runtime)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application):
            if args.nvtx:
                torch.cuda.cudart().cudaProfilerStart()
            try:
                yield
            finally:
                if args.nvtx:
                    torch.cuda.cudart().cudaProfilerStop()
                # This is after requests drain, before engine teardown clears caches.
                engine = engines[0]
                cache_metadata = getattr(engine, "media_preprocess_cache_metadata", None)
                report = {
                    "source_package": prism_infer.__file__,
                    "model": args.model,
                    "options": options,
                    "torch": torch.__version__,
                    "nvtx": args.nvtx,
                    "gpu": torch.cuda.get_device_name(0),
                    "media_preprocess_cache": (
                        None if cache_metadata is None else cache_metadata()
                    ),
                    "metrics": engine.metrics_snapshot(),
                    "cooperative_prefill": engine.cooperative_prefill_policy_metadata(),
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    app.router.lifespan_context = lifespan
    print(
        json.dumps({"source_package": prism_infer.__file__, "options": options}),
        flush=True,
    )
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        if args.pid_file is not None:
            args.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
