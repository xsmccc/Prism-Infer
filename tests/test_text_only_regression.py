"""P2.7 纯文本路径回归验证。"""

from prism_infer import LLM, SamplingParams

from conftest import get_model_path


def test_text_only_generate_greedy_smoke():
    """P2 改动后纯文本 generate 仍能走 engine greedy 路径。"""

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

    print(f"text output token_ids: {outputs[0]['token_ids']}")
    assert len(outputs) == 1
    assert len(outputs[0]["token_ids"]) == 1
    print("text-only engine greedy smoke: PASS")
