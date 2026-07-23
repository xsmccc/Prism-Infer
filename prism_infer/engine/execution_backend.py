"""Startup-selected ModelRunner execution backends.

Preparation and rank-0 sampling are shared, while eager, compile, and CUDA
Graph own distinct forward semantics.  A selected optimized backend never
falls back to another backend at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from prism_infer.config import ExecutionBackendName
from prism_infer.engine.contracts import (
    BatchPhase,
    BatchPlan,
    DeviceBatch,
    ExecutionResult,
    PreparedModelInputs,
)
from prism_infer.engine.compression import compression_supports_cuda_graph
from prism_infer.observability import profile_region
from prism_infer.utils.context import use_context

if TYPE_CHECKING:
    from prism_infer.engine.model_runner import ModelRunner


class ModelExecutionBackend(ABC):
    """Shared tensor preparation and sampling around one forward strategy."""

    name: ExecutionBackendName

    def __init__(self, runner: "ModelRunner") -> None:
        self._runner: ModelRunner | None = runner

    @property
    def runner(self) -> "ModelRunner":
        if self._runner is None:
            raise RuntimeError("execution backend was released")
        return self._runner

    def prepare(self, plan: BatchPlan) -> DeviceBatch:
        runner = self.runner
        seqs = list(plan.sequences)
        if plan.is_prefill:
            vision_patches = plan.num_scheduled_vision_patches
            vision_limit = getattr(
                runner.config,
                "max_vision_patches_per_batch",
                None,
            )
            if (
                vision_limit is not None
                and plan.batch_size > 1
                and vision_patches > int(vision_limit)
            ):
                raise RuntimeError(
                    "prefill plan exceeds the vision patch budget: "
                    f"patches={vision_patches} limit={vision_limit}"
                )
        with profile_region(
            "runner.prepare_inputs",
            metadata={
                "phase": plan.phase.value,
                "vision_patches": (plan.num_scheduled_vision_patches if plan.is_prefill else 0),
            },
        ):
            prepared = (
                runner._prepare_prefill_batch(
                    seqs,
                    prefill_slices=plan.prefill_slices,
                )
                if plan.is_prefill
                else runner._prepare_decode_batch(seqs)
            )
        if not isinstance(prepared, PreparedModelInputs):
            raise TypeError(
                "runner input preparation must return PreparedModelInputs, "
                f"got {type(prepared).__name__}"
            )
        with profile_region("runner.prepare_sample_inputs"):
            sampling_mode = runner.resolve_sampling_mode(seqs) if runner.rank == 0 else None
            temperatures = (
                runner.prepare_sample(seqs)
                if runner.rank == 0 and sampling_mode != "greedy"
                else None
            )
        kv_scale_cache = getattr(runner, "kv_scale_cache", None)
        return DeviceBatch(
            phase=plan.phase,
            sequence_ids=plan.sequence_ids,
            scheduled_token_counts=plan.scheduled_token_counts,
            model_inputs=prepared.model_inputs,
            attention_context=prepared.attention_context,
            temperatures=temperatures,
            execution_bucket=self.execution_bucket(plan),
            sampling_mode=sampling_mode,
            kv_scale_views=(
                () if kv_scale_cache is None else (kv_scale_cache[0], kv_scale_cache[1])
            ),
        )

    def execution_bucket(self, plan: BatchPlan) -> int:
        return plan.batch_size

    def warmup(self, bucket: int | None = None) -> None:
        if bucket is not None and bucket <= 0:
            raise ValueError(f"warmup bucket must be positive, got {bucket}")
        self.runner.warmup_model()

    def capture(self, bucket: int | None = None) -> None:
        raise RuntimeError(f"execution backend {self.name.value!r} does not support capture")

    def execute(self, device_batch: DeviceBatch) -> ExecutionResult:
        runner = self.runner
        if not isinstance(device_batch, DeviceBatch):
            raise TypeError(f"execute requires DeviceBatch, got {type(device_batch).__name__}")
        with use_context(device_batch.attention_context):
            with profile_region("runner.run_model"):
                logits = self.forward_logits(device_batch)
            if runner.rank == 0:
                if (
                    device_batch.sampling_mode != "greedy"
                    and device_batch.temperatures is None
                ):
                    raise RuntimeError("rank 0 non-greedy DeviceBatch requires temperatures")
                with profile_region("runner.sampler"):
                    token_ids = tuple(
                        runner.sampler(
                            logits,
                            device_batch.temperatures,
                            sampling_mode=device_batch.sampling_mode,
                        ).tolist()
                    )
            else:
                token_ids = tuple(None for _ in device_batch.sequence_ids)
        return ExecutionResult(token_ids=token_ids)

    @abstractmethod
    def forward_logits(self, device_batch: DeviceBatch):
        """Execute the backend-specific model path and return logits."""

    def release(self) -> None:
        if self._runner is None:
            return
        runner = self._runner
        self._release_resources(runner)
        self._runner = None

    def _release_resources(self, runner: "ModelRunner") -> None:
        return None


class EagerExecutionBackend(ModelExecutionBackend):
    name = ExecutionBackendName.EAGER

    def forward_logits(self, device_batch: DeviceBatch):
        return self.runner.run_model_eager(
            device_batch.model_inputs,
            is_prefill=device_batch.phase is BatchPhase.PREFILL,
        )


class CompileExecutionBackend(EagerExecutionBackend):
    name = ExecutionBackendName.COMPILE

    def __init__(self, runner: "ModelRunner") -> None:
        super().__init__(runner)
        if runner.config.decode_compile_region != "attention":
            raise ValueError("compile backend requires decode_compile_region='attention'")


class CudaGraphExecutionBackend(ModelExecutionBackend):
    name = ExecutionBackendName.CUDA_GRAPH

    def prepare(self, plan: BatchPlan) -> DeviceBatch:
        fast_batch = self.runner.prepare_single_greedy_decode_cudagraph(plan)
        if fast_batch is not None:
            return fast_batch
        return super().prepare(plan)

    def execute(self, device_batch: DeviceBatch) -> ExecutionResult:
        if not isinstance(device_batch, DeviceBatch):
            raise TypeError(f"execute requires DeviceBatch, got {type(device_batch).__name__}")
        if (
            device_batch.phase is BatchPhase.DECODE
            and device_batch.sampling_mode == "greedy"
        ):
            runner = self.runner
            with use_context(device_batch.attention_context):
                with profile_region("runner.run_model"):
                    sampled_tokens = runner.run_model_cudagraph(
                        device_batch.model_inputs,
                        return_greedy_tokens=True,
                    )
                with profile_region("runner.sampler"):
                    token_ids = tuple(sampled_tokens.tolist())
            return ExecutionResult(token_ids=token_ids)
        return super().execute(device_batch)

    def execution_bucket(self, plan: BatchPlan) -> int:
        if plan.phase is BatchPhase.PREFILL:
            return plan.batch_size
        return next(size for size in self.runner.graph_bs if size >= plan.batch_size)

    def capture(self, bucket: int | None = None) -> None:
        if bucket is not None and bucket <= 0:
            raise ValueError(f"capture bucket must be positive, got {bucket}")
        self.runner.capture_cudagraph()
        if bucket is not None and bucket not in self.runner.graph_bs:
            raise ValueError(f"capture bucket {bucket} is not in {self.runner.graph_bs}")

    def forward_logits(self, device_batch: DeviceBatch):
        if device_batch.phase is BatchPhase.PREFILL:
            return self.runner.run_model_eager(
                device_batch.model_inputs,
                is_prefill=True,
            )
        if not compression_supports_cuda_graph(device_batch.attention_context.compression_metadata):
            raise RuntimeError(
                "CUDA Graph backend received dynamic compression metadata; "
                "runtime eager fallback is forbidden"
            )
        return self.runner.run_model_cudagraph(device_batch.model_inputs)

    def _release_resources(self, runner: "ModelRunner") -> None:
        for name in (
            "graphs",
            "greedy_graphs",
            "graph_pool",
            "graph_vars",
            "graph_logits",
            "graph_greedy_tokens",
            "_single_greedy_decode_batch_cache",
        ):
            if hasattr(runner, name):
                delattr(runner, name)


class CompileGraphExecutionBackend(CudaGraphExecutionBackend):
    """Capture full decode around stateless compiled projection regions."""

    name = ExecutionBackendName.COMPILE_GRAPH

    def __init__(self, runner: "ModelRunner") -> None:
        super().__init__(runner)
        if runner.config.decode_compile_region != "stateless":
            raise ValueError(
                "compile_graph backend requires decode_compile_region='stateless'"
            )


def create_execution_backend(runner: "ModelRunner") -> ModelExecutionBackend:
    """Construct exactly one backend from the validated startup config."""

    backend = ExecutionBackendName(runner.config.execution_backend)
    implementations = {
        ExecutionBackendName.EAGER: EagerExecutionBackend,
        ExecutionBackendName.COMPILE: CompileExecutionBackend,
        ExecutionBackendName.CUDA_GRAPH: CudaGraphExecutionBackend,
        ExecutionBackendName.COMPILE_GRAPH: CompileGraphExecutionBackend,
    }
    try:
        implementation = implementations[backend]
    except KeyError as exc:
        raise ValueError(f"execution backend {backend.value!r} is not implemented") from exc
    return implementation(runner)
