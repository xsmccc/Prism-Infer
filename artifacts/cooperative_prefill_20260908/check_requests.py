"""Exercise three-request Prefill and cancellation on the real TP1 model.

This is a lifecycle/output check, not a timing or benchmark-quality result.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from prism_infer import LLM, SamplingParams


def run(engine, scenario):
    emitted, finished = {}, {}

    def step():
        result = engine.step_result()
        for request_id, token in zip(
            result.plan.sequence_ids, result.execution.token_ids, strict=True
        ):
            if token is not None:
                emitted.setdefault(request_id, []).append(token)
        for output in result.outputs:
            finished[output.request_id] = list(output.token_ids)

    long_id = engine.add_images_request(
        "Describe this image in detail.",
        [Image.new("RGB", (448, 448), (90, 120, 200))],
        SamplingParams(temperature=0, max_tokens=96, ignore_eos=True),
    )
    for _ in range(3):
        step()
    short_ids = [
        engine.add_images_request(
            "Name the main color. Answer with one word.",
            [Image.new("RGB", (448, 448), color)],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        )
        for color in ((220, 40, 40), (40, 190, 60), (50, 80, 230))
    ]
    original_start = engine._start_cooperative_prefill_if_useful
    if scenario == "atomic":
        engine._start_cooperative_prefill_if_useful = lambda plan: False
    step()
    pending = engine._pending_prefill
    if scenario != "atomic":
        assert pending is not None and pending.plan.batch_size == 3
    cancelled = None
    if scenario.startswith("cancel"):
        if scenario == "cancel_language":
            while pending.runner_handle.model_state is None:
                step()
            step()  # Write one language-layer quantum before dropping the state.
        else:
            step()  # Write one vision-block quantum before dropping the state.
        cancelled = short_ids[0]
        assert engine.cancel_request(cancelled)
        assert engine._pending_prefill is None
    try:
        while not engine.is_finished():
            step()
    finally:
        engine._start_cooperative_prefill_if_useful = original_start
    expected_ids = {long_id, *short_ids} - {cancelled}
    assert set(finished) == expected_ids
    assert all(emitted[request_id] == finished[request_id] for request_id in expected_ids)
    assert cancelled is None or cancelled not in emitted
    return {
        "scenario": scenario,
        "long_output": finished[long_id],
        "short_outputs": [finished.get(request_id) for request_id in short_ids],
        "cancelled_request": cancelled,
        "completed_count": len(finished),
        "stream_matches_final": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    options = json.loads(args.config.read_text(encoding="utf-8"))
    options["enable_prefix_caching"] = False
    engine = LLM(args.model, **options)
    report = {"options": options, "cases": []}
    try:
        for scenario in ("atomic", "cooperative", "cancel_vision", "cancel_language"):
            report["cases"].append(run(engine, scenario))
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reference = report["cases"][0]["short_outputs"]
        report["survivor_first_tokens_match_atomic"] = all(
            all(
                value is None or value == reference[i]
                for i, value in enumerate(case["short_outputs"])
            )
            for case in report["cases"]
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
