# Prism-Infer

Prism-Infer 是一个面向 **Qwen3-VL-8B-Instruct** 的单机多模态推理与视觉 KV
Cache 研究引擎。项目以 nano-vLLM 的轻量框架为起点，自实现 Qwen3-VL
text/vision forward、M-RoPE、DeepStack、Paged KV、调度、CUDA Graph decode、KV
trace 和视觉 KV 物理压缩主路径。

Hugging Face 只承担 tokenizer、processor、配置读取与数值参考，不作为模型 forward
或 engine wrapper。当前仓库是研究原型，不是带 HTTP/gRPC 接口的生产 serving 系统。

## 当前能力与状态

| 能力 | 状态 | 当前边界 |
|---|:---:|---|
| text、单图、多图、视频、mixed batch | 已验证 | Qwen3-VL-8B、TP1；覆盖 eager 与 CUDA Graph decode |
| Qwen3-VL full logits / PPL | 已验证 | text 与 VL 路径相对 HF reference 有 strict gate |
| Paged KV、chunked prefill、continuous batching | 已验证 | engine-level arrival/SLO harness，不是网络 server |
| KV trace 与视觉 token 分析 | 已验证 | trace 默认关闭，JSONL 可离线分析 |
| content-aware visual KV physical compaction | 已验证 | BF16、keep=0.5 的 7-image lexical preflight 通过 |
| unit-scale FP8 KV (`fp8_kv`) | 已实现、已拒绝 | direct cast 长输出质量未通过，只保留为失败基线 |
| scaled FP8 KV (`scaled_fp8_kv`) | 正式质量门禁通过 | per-token/per-KV-head scale；allocated KV pool 为 BF16 的 `0.515625x` |
| packed MLP gate/up | 已验证、默认启用 | RTX 5090 TP1；8 个 clean offline cell 的 decode TPOT 改善 `0.483%–0.762%`，不声称稳定 E2E 加速 |
| TP2 | 静态与 IPC preflight 完成 | 动态 correctness/performance 尚无两卡证据 |

权威进度见 [ROADMAP](docs/ROADMAP.md)，允许和禁止使用的结论见
[CLAIMS](docs/CLAIMS.md)，未完成项见 [Known Issues](docs/KNOWN_ISSUES.md)。

## 已验证结果摘要

以下数字都限定于对应 workload 和证据环境，不代表通用模型质量或线上服务性能：

- Qwen3-VL text/vision/M-RoPE/DeepStack/engine 主路径完成模块、full logits、greedy
  与多模态回归门禁。
- last-layer attention visual compaction 在 7 张固定 COCO 图片、35 条 caption、
  output32、BF16、keep=0.5 上，将 physical prompt token 降至 `0.535x`，active
  prompt bytes 降至 `0.538x`；token-F1/ROUGE-L macro drop 分别为
  `0.003288/0.003710`，低于项目预设 `0.01` 门禁。它不是标准 COCO
  CIDEr/SPICE，也不是通用 VQA accuracy。
- 同策略在 COCO batch4/output32 中只有小幅短 workload 收益：decode-step
  `1.021x`、engine output throughput `1.013x`、E2E `1.005x`。
- node-level Systems trace 定位到旧 logits 路径每步把完整 LM head 转为 FP32；改用
  模型精度后，logits CUDA median 从 `4.068 ms` 降至 `0.762 ms`，五类 workload
  TPOT 提升 `1.216x–1.280x`，torch allocator peak 减少 `2,230–2,317 MiB`。
- 同条件 best-stable CUDA Graph 对比中，quality-qualified Prism compact TPOT 仍为
  vLLM 的 `1.34x–1.40x`，即 Prism **尚未反超** vLLM。
- P7.3 的 9-cell engine-level online matrix 中，已完成请求均满足各 cell 预先声明的
  SLO；该结果没有 HTTP/gRPC 开销，也没有同条件 vLLM online goodput 对比。
- packed gate/up 将 single-image Graph replay 的 linear kernels 从 `253` 降到 `217`、
  总 kernels 从 `2,000` 降到 `1,964`；text、单/多图、视频、mixed 与 7-image COCO
  共 8 个 clean cell 均 token exact，unprofiled decode TPOT 改善 `0.483%–0.762%`。
  vision prefill仍有双峰，因此不把该结果扩写成稳定 E2E latency speedup。
- P9-C 的 `scaled_fp8_kv` 在冻结的 DocVQA、MuirBench、MVBench
  development/final 六个正式 cell 中，相对 Prism BF16 均通过 non-inferiority gate；
  allocated KV pool 从 `1,509,949,440` B 降到 `778,567,680` B，节省
  `48.4375%`。这不包含跨框架统一的 page-table/Python allocator 字节。
- 同 logical capacity、同 `0.515625x` allocated-KV-pool 比例下，vLLM 0.24.0
  per-token-head FP8 在 DocVQA/MuirBench 通过、MVBench development/final 未通过
  预注册稳定性门禁；其 MVBench accuracy 点估计反而更高。因此当前外部质量矩阵结论是
  **MIXED**，不是“Prism accuracy 显著高于 vLLM”，也不是完整物理显存 Pareto 胜出。

完整口径、环境和 raw evidence 路径见 [PERFORMANCE_REPORT](docs/PERFORMANCE_REPORT.md)。

## 架构

```text
text / image / images / video
        │
        ├─ HF tokenizer / processor（输入边界）
        ▼
Prism VL inputs + 3D position ids / M-RoPE
        ▼
Vision Encoder ── DeepStack features ── Qwen3-VL decoder
        ▼                                │
Request FSM → Scheduler → BatchPlan → Paged KV manager
                                         │
                   eager / CUDA Graph / Triton paged decode
                                         │
                   logits → sampler → metrics / KV trace
```

主要目录：

```text
prism_infer/
  engine/       # Request/Scheduler/Executor、Paged KV、online metrics、compression
  models/       # Qwen3-VL language model、DeepStack、3D position ids
  vision/       # Vision Encoder、PatchMerger、M-RoPE
  layers/       # attention、linear、norm、sampler
  ops/          # paged decode 与 KV compaction Triton kernels
  analysis/     # KV trace、quality/performance summaries
benchmarks/     # internal、online、external 与 kernel harness
scripts/        # 环境检查、trace、分析与汇总入口
tests/          # 模块、full model、engine、kernel、compression 回归
docs/           # 路线图、验证合同、报告、claim ledger
```

## 环境要求

正式结果的已验证环境：

```text
GPU: NVIDIA GeForce RTX 5090 32 GB
CUDA: 12.8
Python: 3.12.3
PyTorch: 2.6.0a0+ecf3bae40a.nv25.01
Transformers: 5.13.0
Model revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
Model dtype / TP: BF16 / 1
```

项目元数据支持 Python `3.10–3.12`，但上述组合是当前完整门禁环境。完整 8B
formal matrix 的 torch allocator peak 约为 `17.4–17.5 GiB`；建议至少 24 GiB
显存，并在正式 benchmark 前保证 GPU 独占、空闲显存不少于 18 GiB。不同 PyTorch、
CUDA、FlashAttention 或 Triton 组合必须重新做 correctness gate。

## 安装

先按硬件和 CUDA 版本安装匹配的 PyTorch。不要让项目安装过程用任意最新版 Triton
覆盖 PyTorch 自带的绑定版本；FlashAttention 也是可选加速后端，应使用平台兼容的
wheel 或源码构建。

```bash
git clone https://github.com/xsmccc/Prism-Infer.git
cd Prism-Infer

python3.12 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install --upgrade pip

# 先安装与本机 CUDA 匹配的 torch，再安装 Prism 的 Python 依赖。
python -m pip install -e .
```

不加载模型的安装检查：

```bash
python scripts/check_environment.py
```

CPU/SDPA fallback 用于 correctness；正式报告中的 CUDA 路径要求可导入 Triton，已测
prefill backend 还包含平台适配的 FlashAttention。检查脚本会分别报告这两个可选
backend，不会把缺失静默写成性能通过。

## 准备模型

下载完整的 Qwen3-VL-8B-Instruct 本地 snapshot。目录至少应包含：

```text
config.json
tokenizer_config.json
preprocessor_config.json
model.safetensors.index.json
model-*.safetensors
```

设置离线路径并在加载 17 GB 权重前检查模型身份和显存：

```bash
export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct/snapshot
export HF_HUB_OFFLINE=1

python scripts/check_environment.py \
  --model "$PRISM_MODEL_PATH" \
  --require-cuda \
  --min-free-gib 18
```

PASS 输出会包含 Python/Torch/Transformers、可选 backend、GPU free/total memory、
模型类型、权重文件数和 `config.json` SHA256。该检查不加载权重，不等价于 full
model correctness。

## 最小运行

仓库中的 `example.py` 会生成一张 deterministic 448×448 图片并执行 8-token greedy：

```bash
python example.py
```

输出格式：

```text
Token IDs: [<up to 8 integer token ids>]
Text: '<decoded text>'
```

token 内容依赖模型 snapshot 与数值栈；复现时应保存 token IDs、commit、模型 hash
和环境，而不是只比较自然语言字符串。

直接使用 API：

```python
import os

from PIL import Image
from prism_infer import LLM, SamplingParams

image = Image.new("RGB", (448, 448), color=(70, 120, 210))
llm = LLM(
    os.environ["PRISM_MODEL_PATH"],
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    max_num_seqs=1,
    enable_chunked_prefill=False,
)
try:
    result = llm.generate_vl(
        "Describe this image in one short sentence.",
        image,
        SamplingParams(temperature=0.0, max_tokens=8),
        use_tqdm=False,
    )
finally:
    llm.exit()

print(result["token_ids"])
print(result["text"])
```

## 运行质量合格的视觉 KV 压缩路径

当前质量合格配置是 BF16 physical compaction、last-layer attention scorer、
`keep_ratio=0.5`、`min_keep_tokens=32`：

```python
llm = LLM(
    os.environ["PRISM_MODEL_PATH"],
    compression_mode="visual_compact",
    visual_pruning_strategy="attention",
    visual_pruning_attention_last_n_layers=1,
    visual_pruning_keep_ratio=0.5,
    visual_pruning_min_keep_tokens=32,
    enforce_eager=False,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    max_num_seqs=1,
)
```

这组参数只复现已验证候选，不保证任意数据集保持质量。必须同时跑 dense baseline、
保存 token/任务指标和 physical KV 字段。完整成对命令见
[REPRODUCIBILITY](docs/REPRODUCIBILITY.md)。

## 运行质量合格的 scaled FP8 KV 路径

`scaled_fp8_kv` 与旧 `fp8_kv` 是两个独立模式。前者为每个 token、每个 KV head
分别保存 K/V FP32 scale，并把 scale 与 payload 一起纳入 store、paged decode、
copy-on-write、swap、physical compaction 和 CUDA Graph 生命周期：

```python
llm = LLM(
    os.environ["PRISM_MODEL_PATH"],
    compression_mode="scaled_fp8_kv",
    enforce_eager=False,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    max_num_seqs=1,
)
```

正式 PASS 只覆盖冻结的 Qwen3-VL-8B、单卡环境和 P9 质量协议。它不自动证明
`visual_compact_scaled_fp8` 组合、任意模型、任意长上下文或吞吐性能合格。

## KV Trace 与离线分析

KV trace 默认关闭。显式运行三类 deterministic 样例：

```bash
python scripts/run_kv_trace_samples.py \
  --model "$PRISM_MODEL_PATH" \
  --output-dir data/kv_trace_samples \
  --max-tokens 2
```

分析单个 JSONL：

```bash
python scripts/analyze_kv_trace.py \
  data/kv_trace_samples/single_image_description.jsonl \
  --summary-json data/kv_trace_samples/single_image_description.summary.json \
  --markdown data/kv_trace_samples/single_image_description.summary.md \
  --svg data/kv_trace_samples/single_image_description.summary.svg

python scripts/score_visual_tokens.py \
  data/kv_trace_samples/single_image_description.jsonl \
  --output-json data/kv_trace_samples/single_image_description.importance.json \
  --markdown data/kv_trace_samples/single_image_description.importance.md
```

`data/` 默认 gitignored；正式交付必须在报告中记录生成命令、commit 和 raw evidence
路径。

## 验证

不需要模型权重或大显存的安装/CPU smoke：

```bash
python -m compileall prism_infer tests benchmarks scripts
python -m pytest -q \
  tests/test_check_environment.py \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_visual_importance_scoring.py \
  tests/test_compression_off.py \
  tests/test_engine_contracts.py
```

完整模型回归需要本地权重、兼容 CUDA 栈和独占 GPU：

```bash
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" HF_HUB_OFFLINE=1 \
python -m pytest -q tests -s
```

不能用 CPU smoke 替代 full logits、E2E、kernel correctness 或性能门禁。各层 PASS
标准与推荐窄回归见 [VERIFICATION](docs/VERIFICATION.md)。

## 文档导航

- [技术报告](docs/TECHNICAL_REPORT.md)：模型、engine、KV 分析、压缩与系统优化总结。
- [复现实验](docs/REPRODUCIBILITY.md)：从安装 smoke 到正式 GPU matrix 的命令与样例。
- [Known Issues](docs/KNOWN_ISSUES.md)：当前 blocker、限制、恢复条件和待补命令。
- [投递与面试材料](docs/APPLICATION_MATERIALS.md)：可核查项目描述、简历 bullet 与问答。
- [P8 Gate Review](docs/P8_DELIVERY.md)：安装、fresh 8B demo、完整回归与动态性能验收。
- [路线图](docs/ROADMAP.md)：阶段状态与下一执行顺序。
- [验证合同](docs/VERIFICATION.md)：correctness、quality、performance 门禁。
- [性能报告](docs/PERFORMANCE_REPORT.md)：benchmark contract、结果和 raw evidence。
- [Claim Ledger](docs/CLAIMS.md)：允许、必须限定和禁止使用的结论。
- [压缩报告](docs/COMPRESSION_REPORT.md) / [KV 分析报告](docs/KV_ANALYSIS_REPORT.md)。

## 明确不声称

- 不声称 Prism 全面超过 vLLM/SGLang。
- 不声称 visual compaction 让整张 GPU 或整个模型显存减半。
- 不声称 unit-scale `fp8_kv` 已通过质量门禁，也不把 scaled-FP8 的限定结果泛化为
  “所有 FP8 都质量无损”。
- 不声称已完成跨框架 page-table/allocator 全口径物理显存 Pareto，或 scaled-FP8
  已带来正式 runtime speedup。
- 不把 offline output tok/s 当作 online serving goodput。
- 不把 packed MLP 的小幅 decode TPOT 收益写成 online goodput或稳定 E2E 加速。
- 不声称已经验证 TP2、HTTP/gRPC、megakernel、PD 分离或投机解码。
- 不声称 NVFP4 或权重/激活量化已经实现、验证或优于 BF16。

## Acknowledgements

- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的轻量 engine 起点。
- vLLM / PagedAttention 的系统设计启发。
- FlashAttention 与 Triton 的高性能 attention/kernel 生态。

## License

MIT，见 [LICENSE](LICENSE)。
