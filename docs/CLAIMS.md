# Claim Boundaries

This ledger separates implemented mechanisms from measured advantages.

## Supported statements

| Statement | Required scope |
|---|---|
| Prism implements the Qwen3-VL text/vision, M-RoPE, DeepStack, paged KV, scheduling, and native serving path | Qwen3-VL-8B, current repository |
| Compiler/Graph decode has lower TPOT than the tested vLLM/SGLang baselines | RTX 5090, Qwen3-VL-8B, TP1, batch1, greedy output128, frozen H1/H2 |
| Scaled-FP8 reduces allocated KV by 48.44% | Same logical KV capacity; includes FP8 payload and FP32 token-head scales |
| Scaled-FP8 reduces process peak by 8.24% | Prism BF16 versus Prism scaled-FP8 same-capacity NVML runs |
| A similar 4 GiB KV budget increases token capacity by 94.69% | 28,928 to 56,320 KV tokens; this is not measured concurrency |
| Visual compaction releases physical pages | Frozen compaction layouts with page-table rewrite and page-pool reuse |
| Content-addressed reuse works across freshly decoded but byte-identical media objects | One engine process; supported media types; exact model/processor/media/prompt identity |
| Same media with a different question reuses only media-invariant work and the exact visual prefix | Question changes after the last visual placeholder; exact template rebind or fail closed |
| Prism reaches vLLM parity at high media repeat and beats the available SGLang reference | Frozen n60 H3, 75%–100% repeat; vLLM difference 0.28%–0.37%; scoped SGLang reference |
| Qwen3-VL TP2 shards language weights, KV heads, and vocabulary and accelerates the measured decode cells | One host, 2x RTX 5090, BF16, exact greedy output; final versus original correct TP2 Graph: single-image -29.53%, mixed-batch -24.68% TPOT |
| Prism TP2 has lower TPOT than the tested vLLM TP2 baseline | Single 210-token image prompt, batch1, greedy output32, warmup1/repeat3, 2x RTX 5090; 5.970135 versus 6.161151 ms (-3.10%); exact token IDs |

## Statements that require an explicit limitation

- **“600/600 SLO and 241.428 tok/s”** refers to the 100%-repeat H3 workload,
  not unique-media serving.
- **“Avoided 5.55 GiB copy traffic”** is accumulated logical page-copy traffic,
  not a 5.55 GiB reduction in process memory.
- **“Visual tokens were reduced”** means physical prompt KV rows/pages were
  compacted; it does not imply the model weights or full GPU footprint shrink
  by the same ratio.
- **“FP8 passed quality”** refers only to per-token/per-KV-head scaled-FP8 in
  the frozen DocVQA/MuirBench/MVBench protocol. Unit-scale FP8 failed.
- **“CUDA Graph accelerates decode”** means Prism internal eager-to-Graph
  improvement and the frozen external cells; it is not a cross-model result.
- **“Native network serving”** means the local Prism JSON/SSE interface with a
  single engine owner, not OpenAI compatibility or multi-node production.
- **“TP2 cuts memory by about 46%”** means Torch peak allocated per rank. The
  single-image two-rank sum is 6.85% above TP1; aggregate GPU memory is not
  reduced.
- **“TP2 accelerates decode”** applies to the frozen single-image and mixed
  cells. In the external TP2 cell Prism still has worse TTFT/E2E than vLLM
  because vision remains replicated and the measured host has no direct CUDA
  P2P/NVLink path.
- **“Prism matches SGLang TP2”** means TPOT is numerically tied in one cell.
  SGLang diverges at output token 21, so this is not an exact-output win.

## Unsupported statements

Do not claim:

- Prism universally or comprehensively outperforms vLLM and SGLang;
- Prism wins unique-media, cold-start, arbitrary-batch, or arbitrary online
  workloads;
- KV quantization halves total GPU memory;
- scaled-FP8 is faster than Prism BF16;
- capacity improvement is equivalent to measured concurrency or Goodput;
- TP2 universally improves TTFT/E2E, reduces aggregate memory, scales across
  nodes, supports pipeline parallelism, or beats external TP2 engines outside
  the frozen single-image TPOT cell;
- Dynamic Vision Tensor Graph is safe for mixed-shape loaded serving;
- prefix cache state is shared across processes, nodes, or tenants;
- speculative decoding, prefill/decode disaggregation, or a megakernel is
  implemented; or
- a stable token hash alone proves semantic correctness.

## Current bottleneck

At 0%–50% media repeat, vLLM remains faster. The next defensible research
problem is cold multimodal prefill and its interference with decode cadence,
not another cache-only optimization.
