"""Text-only engine generation coverage."""

import pytest
from conftest import get_model_path

from prism_infer import LLM, SamplingParams

pytestmark = [
    pytest.mark.model,
    pytest.mark.gpu,
    pytest.mark.integration,
]


def test_text_only_generate_greedy():
    """The engine returns one greedy token for a text-only request."""

    model_path = get_model_path()
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        compression_mode="off",
    )
    try:
        outputs = llm.generate(
            [[151644, 872, 198, 77091, 198]],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
    finally:
        llm.exit()

    assert len(outputs) == 1
    assert len(outputs[0]["token_ids"]) == 1
