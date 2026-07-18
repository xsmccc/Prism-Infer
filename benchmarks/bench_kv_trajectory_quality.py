"""Compare engine decode logits on one fixed autoregressive trajectory.

This is a P9-C model-precision preflight, not the standard dataset quality
gate.  The first mode generates a greedy trajectory.  Every candidate then
receives exactly those tokens through a forced sampler while its unmodified
decode logits, natural argmax tokens, NLL, perplexity and distribution drift
are recorded.  Fixing the history prevents an early token divergence from
turning all later logits into an incomparable trajectory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_infer import LLM, SamplingParams
from prism_infer.engine.kv_quantization import kv_cache_storage_bytes


TRAJECTORY_SCHEMA_VERSION = 1
TEXT_PROMPT_IDS = [151644, 872, 198, 77091, 198]


@dataclass(frozen=True)
class Trajectory:
    token_ids: list[int]
    natural_argmax_ids: list[int]
    logits: torch.Tensor


class RecordingTrajectorySampler:
    """Record logits and optionally force a preselected single-request path."""

    def __init__(self, forced_token_ids: list[int] | None = None) -> None:
        self.forced_token_ids = (
            None if forced_token_ids is None else list(forced_token_ids)
        )
        self.logits: list[torch.Tensor] = []
        self.natural_argmax_ids: list[int] = []

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        del temperatures
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(
                "trajectory sampler supports exactly one request, got "
                f"logits shape={list(logits.shape)}"
            )
        step = len(self.logits)
        snapshot = logits[0].detach().float().cpu().contiguous()
        natural_token = int(snapshot.argmax().item())
        self.logits.append(snapshot)
        self.natural_argmax_ids.append(natural_token)
        if self.forced_token_ids is None:
            selected_token = natural_token
        else:
            if step >= len(self.forced_token_ids):
                raise RuntimeError(
                    "engine requested more trajectory steps than forced tokens"
                )
            selected_token = self.forced_token_ids[step]
        return torch.tensor(
            [selected_token],
            dtype=torch.long,
            device=logits.device,
        )

    def finish(self, emitted_token_ids: list[int]) -> Trajectory:
        if not self.logits:
            raise RuntimeError("trajectory sampler did not observe any logits")
        if len(emitted_token_ids) != len(self.logits):
            raise RuntimeError(
                "emitted token/logit step mismatch: "
                f"tokens={len(emitted_token_ids)}, logits={len(self.logits)}"
            )
        if (
            self.forced_token_ids is not None
            and emitted_token_ids != self.forced_token_ids
        ):
            raise RuntimeError("engine output did not follow the forced trajectory")
        if (
            self.forced_token_ids is None
            and emitted_token_ids != self.natural_argmax_ids
        ):
            raise RuntimeError(
                "baseline engine output did not follow the natural greedy trajectory"
            )
        return Trajectory(
            token_ids=list(emitted_token_ids),
            natural_argmax_ids=list(self.natural_argmax_ids),
            logits=torch.stack(self.logits),
        )


def _stable_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def compare_trajectories(
    baseline: Trajectory,
    candidate: Trajectory,
) -> dict[str, object]:
    """Return distribution and target-likelihood drift on a fixed history."""

    if baseline.token_ids != candidate.token_ids:
        raise ValueError("candidate must use the baseline token trajectory")
    if baseline.logits.shape != candidate.logits.shape:
        raise ValueError(
            "trajectory logit shapes differ: "
            f"{list(baseline.logits.shape)} vs {list(candidate.logits.shape)}"
        )
    targets = torch.tensor(baseline.token_ids, dtype=torch.long)
    baseline_logits = baseline.logits.float()
    candidate_logits = candidate.logits.float()
    difference = (candidate_logits - baseline_logits).abs()
    baseline_nll = F.cross_entropy(baseline_logits, targets)
    candidate_nll = F.cross_entropy(candidate_logits, targets)
    baseline_ppl = torch.exp(baseline_nll)
    candidate_ppl = torch.exp(candidate_nll)
    baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
    candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
    baseline_probs = baseline_log_probs.exp()
    kl_divergence = (
        baseline_probs * (baseline_log_probs - candidate_log_probs)
    ).sum(dim=-1)
    natural_matches = sum(
        baseline_token == candidate_token
        for baseline_token, candidate_token in zip(
            baseline.natural_argmax_ids,
            candidate.natural_argmax_ids,
        )
    )
    return {
        "steps": len(baseline.token_ids),
        "logits_shape": list(baseline.logits.shape),
        "baseline_logits_sha256": _tensor_sha256(baseline.logits),
        "candidate_logits_sha256": _tensor_sha256(candidate.logits),
        "max_abs_logit_diff": float(difference.max().item()),
        "mean_abs_logit_diff": float(difference.mean().item()),
        "baseline_nll": float(baseline_nll.item()),
        "candidate_nll": float(candidate_nll.item()),
        "nll_delta": float((candidate_nll - baseline_nll).item()),
        "baseline_ppl": float(baseline_ppl.item()),
        "candidate_ppl": float(candidate_ppl.item()),
        "ppl_ratio": float((candidate_ppl / baseline_ppl).item()),
        "mean_baseline_to_candidate_kl": float(kl_divergence.mean().item()),
        "max_baseline_to_candidate_kl": float(kl_divergence.max().item()),
        "natural_argmax_matches": natural_matches,
        "natural_argmax_match_ratio": natural_matches / len(baseline.token_ids),
        "natural_argmax_stable_prefix": _stable_prefix(
            baseline.natural_argmax_ids,
            candidate.natural_argmax_ids,
        ),
        "candidate_natural_argmax_ids": candidate.natural_argmax_ids,
    }


def aggregate_comparisons(
    comparisons: list[dict[str, object]],
) -> dict[str, int | float]:
    """Aggregate equal-vocabulary trajectory metrics with step weighting."""

    if not comparisons:
        raise ValueError("at least one trajectory comparison is required")
    total_steps = sum(int(row["steps"]) for row in comparisons)
    if total_steps <= 0:
        raise ValueError("trajectory comparisons must contain positive steps")

    def weighted_mean(key: str) -> float:
        return sum(
            float(row[key]) * int(row["steps"])
            for row in comparisons
        ) / total_steps

    natural_matches = sum(
        int(row["natural_argmax_matches"]) for row in comparisons
    )
    return {
        "cases": len(comparisons),
        "steps": total_steps,
        "max_abs_logit_diff": max(
            float(row["max_abs_logit_diff"]) for row in comparisons
        ),
        "step_weighted_mean_abs_logit_diff": weighted_mean(
            "mean_abs_logit_diff"
        ),
        "step_weighted_mean_baseline_to_candidate_kl": weighted_mean(
            "mean_baseline_to_candidate_kl"
        ),
        "step_weighted_mean_nll_delta": weighted_mean("nll_delta"),
        "step_weighted_mean_ppl_ratio": weighted_mean("ppl_ratio"),
        "natural_argmax_matches": natural_matches,
        "natural_argmax_match_ratio": natural_matches / total_steps,
        "minimum_case_stable_prefix": min(
            int(row["natural_argmax_stable_prefix"]) for row in comparisons
        ),
    }


def _images_and_video() -> tuple[Image.Image, Image.Image, list[Image.Image]]:
    image_a = Image.new("RGB", (448, 448), color=(100, 150, 200))
    image_b = Image.new("RGB", (448, 448), color=(200, 120, 80))
    frames = [
        Image.new("RGB", (448, 448), color=(80 + index * 30, 120, 180))
        for index in range(4)
    ]
    return image_a, image_b, frames


def _cases() -> list[dict[str, Any]]:
    image_a, image_b, frames = _images_and_video()
    return [
        {"name": "text", "type": "text", "prompt": TEXT_PROMPT_IDS},
        {
            "name": "single_image",
            "type": "image",
            "prompt": "Describe this image.",
            "image": image_a,
        },
        {
            "name": "multi_image",
            "type": "images",
            "prompt": "Compare these images.",
            "images": [image_a, image_b],
        },
        {
            "name": "video",
            "type": "video",
            "prompt": "Describe this video.",
            "video": frames,
        },
    ]


def _run_case(
    llm: LLM,
    case: dict[str, Any],
    sampling: SamplingParams,
    forced_token_ids: list[int] | None,
) -> Trajectory:
    sampler = RecordingTrajectorySampler(forced_token_ids)
    llm.model_runner.sampler = sampler
    request_type = case["type"]
    if request_type == "text":
        output = llm.generate([case["prompt"]], sampling, use_tqdm=False)[0]
    elif request_type == "image":
        output = llm.generate_vl(
            case["prompt"], case["image"], sampling, use_tqdm=False
        )
    elif request_type == "images":
        output = llm.generate_images(
            case["prompt"], case["images"], sampling, use_tqdm=False
        )
    elif request_type == "video":
        output = llm.generate_video(
            case["prompt"], case["video"], sampling, use_tqdm=False
        )
    else:
        raise ValueError(f"unsupported trajectory case type: {request_type!r}")
    return sampler.finish(list(output["token_ids"]))


def _cache_record(llm: LLM) -> dict[str, object]:
    payload = llm.model_runner.kv_cache
    scales = llm.model_runner.kv_scale_cache
    storage = kv_cache_storage_bytes(payload, scales)
    return {
        "payload_dtype": str(payload.dtype),
        "payload_shape": list(payload.shape),
        "scale_dtype": "none" if scales is None else str(scales.dtype),
        "scale_shape": [] if scales is None else list(scales.shape),
        "payload_bytes": storage.payload,
        "scale_bytes": storage.scales,
        "total_bytes": storage.total,
    }


def _build_llm(args: argparse.Namespace, mode: str) -> LLM:
    return LLM(
        args.model,
        compression_mode=mode,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        num_kvcache_blocks=args.num_kvcache_blocks,
        kvcache_block_size=args.kvcache_block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        logits_precision="model",
    )


def _run_mode(
    args: argparse.Namespace,
    mode: str,
    cases: list[dict[str, Any]],
    baseline: dict[str, Trajectory] | None,
) -> tuple[dict[str, object], dict[str, Trajectory] | None]:
    llm = _build_llm(args, mode)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    trajectories: dict[str, Trajectory] = {}
    case_records: dict[str, object] = {}
    comparisons: list[dict[str, object]] = []
    try:
        cache_record = _cache_record(llm)
        for case in cases:
            name = case["name"]
            forced = None if baseline is None else baseline[name].token_ids
            trajectory = _run_case(llm, case, sampling, forced)
            trajectories[name] = trajectory
            row: dict[str, object] = {
                "token_ids": trajectory.token_ids,
                "natural_argmax_ids": trajectory.natural_argmax_ids,
                "logits_sha256": _tensor_sha256(trajectory.logits),
            }
            if baseline is not None:
                comparison = compare_trajectories(
                    baseline[name], trajectory
                )
                row["comparison_to_baseline"] = comparison
                comparisons.append(comparison)
            case_records[name] = row
        mode_record: dict[str, object] = {
            "mode": mode,
            "kv_cache": cache_record,
            "cases": case_records,
        }
        if comparisons:
            mode_record["aggregate_comparison_to_baseline"] = (
                aggregate_comparisons(comparisons)
            )
        return (
            mode_record,
            trajectories if baseline is None else None,
        )
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _git_metadata() -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            text=True,
        ).strip()
    )
    return {"git_commit": commit, "git_dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--modes",
        default="off,fp8_kv,scaled_fp8_kv",
        help="baseline must be first",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1280)
    parser.add_argument("--num-kvcache-blocks", type=int, default=16)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.max_tokens < 2:
        raise SystemExit("--max-tokens must be >= 2")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes or modes[0] != "off" or len(modes) != len(set(modes)):
        raise SystemExit("--modes must be unique and begin with off")

    cases = _cases()
    records: list[dict[str, object]] = []
    baseline: dict[str, Trajectory] | None = None
    for mode in modes:
        print(f"running trajectory mode={mode}", file=sys.stderr)
        mode_record, new_baseline = _run_mode(
            args,
            mode,
            cases,
            baseline,
        )
        records.append(mode_record)
        if baseline is None:
            if new_baseline is None:
                raise RuntimeError("baseline mode did not return trajectories")
            baseline = new_baseline

    output = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "record_type": "kv_trajectory_quality",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "model_precision_preflight_not_standard_quality_gate",
        "model": str(Path(args.model).resolve()),
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": args.max_tokens,
            "logits_precision": "model",
        },
        "environment": {
            **_git_metadata(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "modes": records,
    }
    serialized = json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"wrote trajectory record to {output_path}", file=sys.stderr)
    print(serialized)


if __name__ == "__main__":
    main()
