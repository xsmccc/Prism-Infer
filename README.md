# Prism-Infer

Prism-Infer 是一个面向 Qwen3-VL-8B-Instruct 的多模态推理引擎研究项目，重点关注视觉 token 在端到端推理、Paged KV Cache、KV 行为分析和压缩策略中的工程实现。

项目从 nano-vLLM 的轻量推理框架出发，但当前目标不是包装 Hugging Face、vLLM 或 SGLang。Hugging Face 主要作为 tokenizer、processor 和数值参考边界；Qwen3-VL 模型前向、M-RoPE、Vision Encoder、DeepStack 注入、engine 调度、KV cache、paged decode、trace 和压缩研究路径都在本仓库内实现和验证。

## 项目定位

- 自实现 Qwen3-VL 多模态推理路径，覆盖 text、image、multi-image、video 和 mixed batch。
- 建立模块级、full logits、端到端 greedy、trace on/off、kernel correctness 等多层验证门禁。
- 以可靠 FP baseline 为基础，分析视觉 token 的 KV/attention 行为，再推进视觉 token KV cache 压缩。
- 所有性能或压缩收益必须来自可复现实测；未验证内容只作为风险或计划记录。

## 核心模块

```text
prism_infer/
  engine/          # LLM engine、scheduler、sequence、block manager、model runner、compression metadata
  models/          # Qwen3-VL text model、DeepStack 注入、3D position ids
  vision/          # Qwen3-VL Vision Encoder、PatchMerger、M-RoPE
  layers/          # attention、linear、layernorm、sampler 等模型层
  ops/             # Triton/CUDA kernel，例如 paged decode attention
  analysis/        # KV trace、visual-token importance 等离线分析工具
scripts/           # trace、analysis、scoring 等命令行脚本
tests/             # 模型、engine、VL、trace、kernel、compression 回归测试
benchmarks/        # paged decode、VL CUDA Graph 等 benchmark
docs/              # 路线图、验证标准、分析报告、问题记录
```

## 快速开始

准备本地 Qwen3-VL-8B-Instruct 模型快照，并设置模型路径：

```bash
cd /data/Prism-Infer
pip install -e .

export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export HF_HUB_OFFLINE=1
```

单图 greedy 生成示例：

```python
from PIL import Image

from prism_infer import LLM, SamplingParams

model_path = "/path/to/Qwen3-VL-8B-Instruct"
image = Image.new("RGB", (448, 448), color=(70, 120, 210))

llm = LLM(
    model_path,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    enable_chunked_prefill=False,
)

out = llm.generate_vl(
    "Describe this image in one short sentence.",
    image,
    SamplingParams(temperature=0.0, max_tokens=8),
    use_tqdm=False,
)
print(out["token_ids"])
print(out["text"])
```

更多多图、视频、mixed batch 和 CUDA Graph decode 示例见 `tests/` 与 `benchmarks/`。

## KV Trace 与分析

KV trace 默认关闭。显式开启后，trace 会写出 JSONL，包含模型配置、batch metadata、text/image/video token span、attention/KV 统计和离线分析所需字段。

运行样例 trace：

```bash
PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
python scripts/run_kv_trace_samples.py \
  --output-dir data/kv_trace_samples \
  --max-tokens 2
```

分析已有 trace：

```bash
python scripts/analyze_kv_trace.py \
  data/kv_trace_samples/single_image_description.jsonl \
  --summary-json data/kv_trace_samples/single_image_description.summary.json \
  --markdown data/kv_trace_samples/single_image_description.summary.md \
  --svg data/kv_trace_samples/single_image_description.summary.svg
```

视觉 token 重要性离线评分：

```bash
python scripts/score_visual_tokens.py \
  data/kv_trace_samples/single_image_description.jsonl \
  --output-json data/kv_trace_samples/single_image_description.importance.json \
  --markdown data/kv_trace_samples/single_image_description.importance.md
```

`data/` 是实验产物目录，默认不入库。

## 验证

权威验证矩阵见 `docs/VERIFICATION.md`。不同层级的验证不能互相替代：语法检查、模块对齐、full logits、端到端 greedy、trace 等价、kernel correctness 和 benchmark 各自对应不同门禁。

轻量检查示例：

```bash
python -m compileall prism_infer tests benchmarks scripts
python -m pytest -q \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_visual_importance_scoring.py \
  tests/test_kv_trace_no_output_change.py \
  tests/test_paged_decode_kernel.py \
  tests/test_compression_off.py
```

重型 Qwen3-VL 验证需要本地模型权重和 GPU 环境：

```bash
PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
HF_HUB_OFFLINE=1 \
python -m pytest -q tests -s
```

## 文档导航

- `docs/ROADMAP.md`: 阶段路线图、当前真实状态、下一步计划。
- `docs/VERIFICATION.md`: 各阶段验证命令、PASS 标准和禁止行为。
- `docs/KV_ANALYSIS_REPORT.md`: P4 KV trace 分析报告。
- `docs/COMPRESSION_REPORT.md`: P5 KV cache 压缩研究报告。
- `docs/ISSUE_LOG.md`: 已定位问题、修复记录和剩余风险。

README 只提供项目介绍和入口；阶段进度、实验数字、压缩收益和未完成风险以 `docs/` 下的专门文档为准。

## Acknowledgements

- Original nano-vLLM by GeeeekExplorer.
- vLLM and PagedAttention design.
- FlashAttention for high-performance attention kernels.

## License

MIT. See `LICENSE`.
