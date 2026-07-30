# Prism-Infer

## P17 内容寻址压缩多模态前缀缓存

P17 将 P16 依赖 Python 对象身份的视觉缓存升级为安全的内容寻址复用：缓存身份覆盖
模型/processor、媒体像素与布局，以及截至最后一个视觉占位符的精确 prompt prefix。
它能跨请求复用 processor 张量、Vision/DeepStack 输出和已经物理压缩的 scaled-FP8
前缀 KV，同时保留 M-RoPE 逻辑位置、页引用、部分尾页 CoW、可复用尾页池和按收益/页
淘汰语义；同一媒体换问题时只复用媒体不变量，不复用问题后缀。

在每个请求都重新创建媒体对象的 60-request 冻结矩阵中，Prism 在 75%/100% 重复率
达到 60/60 SLO，raw throughput 为 `224.279/224.369 tok/s`，与开启官方缓存路径的
vLLM 相差 `0.37%/0.28%`；100% 重复时相对可用的 SGLang cache-on 参考高
`0.82%` raw、`4.30%` Goodput。unique 至 50% 重复时 vLLM 仍领先，因此不声明全面
胜出。最终 n600 100%-repeat 闭环为 `241.428 tok/s`、600/600 SLO，相比 P16
Goodput `+6.68%`，同时保持 H1/H2 exact、scaled-FP8 KV `-48.44%`、视觉物理页
回收和退出显存释放。完整设计、失败实验、公平性边界与复现见
[P17 Content-Addressed Prefix Cache 结果](docs/P17_CONTENT_ADDRESSED_PREFIX_CACHE_RESULTS.md)。

## P16 稳态多模态 Goodput 更新

在 RTX 5090、Qwen3-VL-8B、600-request Poisson rate-4 冻结负载上，P16
将 class-SLO Goodput 从 P16 调度器 cache-off 的 `171.538 tok/s` 提升到
`226.311 tok/s`（`+31.93%`），同一 workload 下高于 vLLM `6.70%`、高于
SGLang `15.01%`；raw throughput 为 `241.184 tok/s`，与两者相差不超过
`0.13%`。600/600 请求完成且无失败，H1/H2 64-token hash exact、
scaled-FP8 KV bytes `-48.44%` 与视觉物理页回收保持不变。

该结果限定于 warm、单进程、重复媒体对象 workload。256 MiB exact LRU 只复用
Vision Encoder 主输出与 DeepStack 输出；语言 prefill、KV 写入、decode、采样和每次
64-token 生成仍完整执行。它不是 unique-media、content-addressed network cache 或
“Prism 全面超过 vLLM/SGLang”的结论。完整协议、公平性边界、失败候选、Profiler 与
面试口径见 [P16 Steady-State Goodput 结果](docs/P16_STEADY_STATE_GOODPUT_RESULTS.md)。

Prism-Infer 是一个面向 **Qwen3-VL-8B-Instruct** 的单机多模态推理与视觉 KV
Cache 研究引擎。项目以 nano-vLLM 的轻量框架为起点，自实现 Qwen3-VL
text/vision forward、M-RoPE、DeepStack、Paged KV、调度、CUDA Graph decode、KV
trace 和视觉 KV 物理压缩主路径。

Hugging Face 只承担 tokenizer、processor、配置读取与数值参考，不作为模型 forward
或 engine wrapper。当前仓库是研究原型，已包含原生 HTTP/SSE 推理边界，但不是
OpenAI-compatible 或多机生产 serving 系统。

> **2026-07-23 correctness 更正：**旧 P9-D/P10 文档中的 H1 输出
> `76ad1f...14c6` 虽然 repeat 内 token hash 一致，但内容与八张输入图无关，不能作为
> 语义正确或外部领先证据。根因是当前 Torch 2.11 环境没有独立 `flash-attn` 时，
> flattened `[tokens, heads, dim]` 被直接传给 SDPA，attention 维度解释错误。
> commit `26deccd` 已改用 vLLM bundled varlen FlashAttention，并加入形状正确的
> per-sequence SDPA fallback。下文的当前三引擎数字均来自修复后的语义正确运行；
> 旧 hash 与旧排名只保留为历史失效记录。

## 当前能力与状态

| 能力 | 状态 | 当前边界 |
|---|:---:|---|
| text、单图、多图、视频、mixed batch | 已验证 | Qwen3-VL-8B、TP1；覆盖 eager 与 CUDA Graph decode |
| Qwen3-VL full logits / PPL | 已验证 | text 与 VL 路径相对 HF reference 有 strict gate |
| Paged KV、chunked prefill、continuous batching | 已验证 | engine-level 与原生 HTTP/SSE arrival/SLO harness |
| 原生 HTTP/SSE serving | 已验证 | JSON/SSE、bounded ingress、断连取消、单 engine owner；不是生产 API |
| KV trace 与视觉 token 分析 | 已验证 | trace 默认关闭，JSONL 可离线分析 |
| 模态自适应 visual KV physical compaction | 标准质量与动态页复用已验证 | image/mixed floor768、video-only floor256、keep0.6；DocVQA/MuirBench/MVBench formal development PASS |
| unit-scale FP8 KV (`fp8_kv`) | 已实现、已拒绝 | direct cast 长输出质量未通过，只保留为失败基线 |
| scaled FP8 KV (`scaled_fp8_kv`) | 质量、Graph、容量、显存已闭环 | per-token/per-KV-head scale；同容量 KV 为 BF16 的 `0.515625x`，同约 4 GiB budget 容量提升 `94.69%` |
| packed MLP gate/up | 已验证、默认启用 | RTX 5090 TP1；8 个 clean offline cell 的 decode TPOT 改善 `0.483%–0.762%`，不声称稳定 E2E 加速 |
| compile + CUDA Graph decode hot path | H1/H2 三引擎闭环 | RTX 5090、TP1、batch1、greedy、output128；Prism BF16 与 scaled-FP8 TPOT 均低于同协议 vLLM/SGLang |
| arrival-driven external H3 | 正式闭环、loaded goodput 未胜出 | 600 requests、Poisson、四类 conditional-video mix；raw throughput 距 vLLM/SGLang 不到 0.8%，但 SLO goodput 明显落后 |
| P15 balanced loaded serving | 四次复测通过 | 60-request frozen trace；deadline-aware prefill coalescing + CPU preprocessing resource budget；TPOT 低于 vLLM/SGLang bounded references，raw/TTFT/Goodput 仍落后 |
| P16 steady-state repeated-media serving | 600-request 正式闭环、限定 Goodput 胜出 | SLO slack/cost 调度 + 256 MiB exact Vision/DeepStack LRU；Goodput `226.311 tok/s`，高于同 workload vLLM/SGLang `6.70%/15.01%`；不外推到 unique-media |
| P17 content-addressed compressed prefix cache | fresh-object 矩阵与 n600 正式闭环 | 内容 SHA256 + media-layout prompt rebind + compacted scaled-FP8 prefix pages/CoW/tail pool；高重复负载与 vLLM 持平并超过可用 SGLang 参考，unique/低重复仍落后 |
| phase-decomposed multimodal prefill | 已实现原型、已拒绝并删除 | H1 单请求 exact 且最大执行段缩短；同 trace loaded goodput/TTFT/TPOT 未通过保留门槛 |
| dynamic-shape Vision tensor Graph | loaded serving 默认关闭 | 新 RTX 5090 的 600-request mixed trace 出现错误 token；保留 decode CUDA Graph |
| vision-aware scheduler | 实验实现、默认关闭 | 有界旁路改善 TTFT/E2E 中位数与尾部，但 loaded SLO goodput 未胜出 |
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
- clean `4779342` 的 H1/H2 三引擎冻结集使用同 GPU、同 prompt-token SHA256、
  warmup2/repeat5、batch1、greedy output128。Prism BF16 的 H1/H2 TPOT 为
  `9.8821/9.8680 ms`，SGLang 为 `10.3520/10.3689 ms`，vLLM 为
  `10.5276/10.5278 ms`；Prism 相对 SGLang 低 `4.54%–4.83%`，相对 vLLM
  低 `6.13%–6.27%`。H1 对 SGLang 的 E2E 只低 `0.07%`，按近似持平报告。
  H2 的 vLLM marker 兼容与 SGLang 16 帧无损媒体输入都经过 prompt/hash 审计。
- P7.3 的 9-cell engine-level online matrix 中，已完成请求均满足各 cell 预先声明的
  SLO；该结果没有 HTTP/gRPC 开销，也没有同条件 vLLM online goodput 对比。
- packed gate/up 将 single-image Graph replay 的 linear kernels 从 `253` 降到 `217`、
  总 kernels 从 `2,000` 降到 `1,964`；text、单/多图、视频、mixed 与 7-image COCO
  共 8 个 clean cell 均 token exact，unprofiled decode TPOT 改善 `0.483%–0.762%`。
  vision prefill仍有双峰，因此不把该结果扩写成稳定 E2E latency speedup。
- P9-C 的 `scaled_fp8_kv` 在冻结的 DocVQA、MuirBench、MVBench
  development/final 六个正式 cell 中，相对 Prism BF16 均通过 non-inferiority gate；
  allocated KV pool 从 `1,509,949,440` B 降到 `778,567,680` B，节省
  `48.4375%`。P10 的同容量进程 NVML 峰值从 `23,938` 降至 `21,966 MiB`，
  实际下降 `8.24%`；同约 4 GiB KV budget 下 capacity 从 `28,928` 增至
  `56,320` tokens，提升 `94.69%`。
- 容量 profile 的 H1/H2 TPOT 为 `10.2363/10.2588 ms`，相对 SGLang 低
  `1.06%–1.12%`、相对 vLLM 低 `2.55%–2.77%`。这说明容量接近翻倍后仍保持
  受限场景 TPOT 优势，不说明 scaled-FP8 比 Prism BF16 更快；E2E 结论为 mixed。
- 同 logical capacity、同 `0.515625x` allocated-KV-pool 比例下，vLLM 0.24.0
  per-token-head FP8 在 DocVQA/MuirBench 通过、MVBench development/final 未通过
  预注册稳定性门禁；其 MVBench accuracy 点估计反而更高。因此当前外部质量矩阵结论是
  **MIXED**，不是“Prism accuracy 显著高于 vLLM”，也不是完整物理显存 Pareto 胜出。
- P11 的模态自适应组合策略使用 `keep=0.6`、image/mixed floor768、video-only
  floor256。DocVQA/MuirBench/MVBench development 分别为 `200/200/97` 个 paired
  样本，三项 formal gate 全 PASS；MVBench 97/97 output token exact。
- H1 batch2 仅给 11 个 KV pages 时，dense 路径只能 batch1 decode；compact
  路径把每请求 prompt 从 7 页降到 4 页，第一页表释放的 `[1,0,5]` 被第二请求
  prefill 复用，378/384 decode steps 进入 batch2。该容量受限 cell 的 requests/s
  提升 `58.83%`，不外推为通用 online goodput。
- 重型 Vision tensor CUDA Graph 在 clean H1 中保持 token exact，engine TTFT
  从 `244.035` 降至 `229.270 ms`（-6.05%）；H2 未观察到可靠加速，单图默认
  fallback eager。
- P12 的 600-request rate-4 conditional-video H3 中，vLLM/SGLang/Prism raw
  throughput 为 `241.489/241.447/239.607 tok/s`；Prism 在约相同 4 GiB KV
  budget 下保留 `56,320` tokens，约为两家 BF16 pool 的 `1.93x/1.95x`，但
  class-aware goodput 仅 `65.093 tok/s`，明显低于 `212.108/196.779 tok/s`。
  该结果说明容量与 loaded token cadence 是不同瓶颈，不构成 online 胜出。
- P13 实现了独立 VISION、缓存 visual/DeepStack embedding、chunked language
  prefill 与 BF16 prefill workspace 原型。H1 1024 chunk 保持 64-token exact，
  将 mixed trace 的 prefill max 从 `446.229` 降至 `119.489 ms`，但同 trace
  class-aware goodput 从 `21.569` 降至 `14.197 tok/s`（-34.18%），TTFT p50
  `+16.5%`、TPOT p50 `+1.98%`，所以候选代码已删除，只保留失败证据。
- P15 在同一 RTX 5090、Qwen3-VL-8B 与冻结 60-request loaded trace 上，将
  underfilled prefill 设为 250 ms deadline-aware coalescing，并把在线 CPU intra-op
  资源从默认 104 线程显式限制为 8，避免媒体预处理饿死 CUDA Graph 提交。四次
  中位数为 `215.628 tok/s`、TTFT `776.863 ms`、TPOT `12.490 ms`、
  class-SLO Goodput `75.566 tok/s`；TPOT 相对 vLLM/SGLang bounded references
  低 `8.56%/13.86%`，但另外三项仍未超过外部系统。H1/H2 64-token hash exact，
  KV bytes `-48.44%` 与视觉物理页回收保持不变。
- P16 在冻结 600-request trace 上先用 latest-start/cost-aware 调度把 Goodput 从
  P15 n600 的 `67.427` 提升至 cache-off 的 `171.538 tok/s`，再利用精确、受限的
  视觉编码结果复用达到 `226.311 tok/s`（563/600 请求同时满足 class TTFT/TPOT
  SLO）。相同 workload 下 vLLM/SGLang 为 `212.108/196.779 tok/s`；P16 n600
  TPOT `14.329 ms` 与 vLLM 近似持平、低于 SGLang `0.44%`，因此 headline 是
  repeated-media SLO Goodput 而不是全面 TPOT 排名。
- P17 用模型/processor/媒体内容/布局/prompt-prefix SHA256 取代对象身份作为缓存
  语义，并把复用扩展到物理压缩的 scaled-FP8 prefix KV。fresh-object n60 的
  100% 重复与同媒体不同问题分别为 `224.369/224.301 tok/s`、均 60/60 SLO；
  n600 为 `241.428 tok/s`、600/600 SLO。它在高重复负载超过可用 SGLang
  cache-on 参考并与 vLLM 相差 0.3% 以内，但 unique/低重复仍由 vLLM 领先。

最终口径、环境和 raw evidence 路径见
[P10 最终结果](docs/P10_FINAL_RESULTS.md)、
[P11 结果](docs/P11_MULTIMODAL_COMPACTION_RESULTS.md) 与
[P12 Online 结果](docs/P12_ONLINE_GOODPUT_RESULTS.md)、
[P13 Phase Prefill 结果](docs/P13_PHASE_DECOMPOSED_PREFILL_RESULTS.md) 与
[P14 Loaded Decode 结果](docs/P14_LOADED_DECODE_RESULTS.md)、
[P15 Balanced Serving 结果](docs/P15_BALANCED_MULTIMODAL_RESULTS.md) 与
[P16 Steady-State Goodput 结果](docs/P16_STEADY_STATE_GOODPUT_RESULTS.md)、
[P17 Content-Addressed Prefix Cache 结果](docs/P17_CONTENT_ADDRESSED_PREFIX_CACHE_RESULTS.md)、
[Network Serving 结果](docs/NETWORK_SERVING_RESULTS.md)、
[PERFORMANCE_REPORT](docs/PERFORMANCE_REPORT.md)。

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
P10/P11 GPU UUID: GPU-7f63f8b0-1027-d3bf-18b7-5102cbc9f2eb
P15/P16/P17 GPU UUID: GPU-a0340044-fe48-ceca-08e0-a50d9bcdd79a
Driver: 580.105.08
CUDA: 13.0
Python: 3.12.3
PyTorch: 2.11.0+cu130
Transformers: 5.14.1
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
- [P10 最终结果](docs/P10_FINAL_RESULTS.md)：compile/Graph H1/H2 外部对比与 scaled-FP8 KV 显存/容量 Pareto。
- [P11 结果](docs/P11_MULTIMODAL_COMPACTION_RESULTS.md)：Vision Graph、模态自适应视觉 KV 正式质量与动态页复用。
- [P12 Online 结果](docs/P12_ONLINE_GOODPUT_RESULTS.md)：600-request 多模态 arrival/SLO goodput 与 vLLM/SGLang 固定协议对比。
- [P13 Phase Prefill 结果](docs/P13_PHASE_DECOMPOSED_PREFILL_RESULTS.md)：可调度多模态 prefill 原型、correctness 问题、loaded 否决与删除。
- [P14 Loaded Decode 结果](docs/P14_LOADED_DECODE_RESULTS.md)：block/layer cooperative prefill、B1--B8 Graph 与 guarded FP8 LM head。
- [P15 Balanced Serving 结果](docs/P15_BALANCED_MULTIMODAL_RESULTS.md)：deadline-aware coalescing、CPU launch starvation 根因与四次 loaded 复测。
- [P16 Steady-State Goodput 结果](docs/P16_STEADY_STATE_GOODPUT_RESULTS.md)：600-request SLO 调度、exact Vision/DeepStack LRU、外部 Goodput 对比、Profiler、失败候选与面试边界。
- [P17 Content-Addressed Prefix Cache 结果](docs/P17_CONTENT_ADDRESSED_PREFIX_CACHE_RESULTS.md)：fresh-object 重复率矩阵、安全内容指纹、媒体换问题复用、压缩前缀页/CoW/尾页池、公平 cache-on 对比与 n600 闭环。
- [秋招最终交付](docs/FINAL_DELIVERY.md)：项目定位、最终数字、简历 bullets、面试主线和交付边界。
- [Claim Ledger](docs/CLAIMS.md)：允许、必须限定和禁止使用的结论。
- [压缩报告](docs/COMPRESSION_REPORT.md) / [KV 分析报告](docs/KV_ANALYSIS_REPORT.md)。

## 明确不声称

- 不声称 Prism 全面超过 vLLM/SGLang。
- 不声称 visual compaction 让整张 GPU 或整个模型显存减半。
- 不声称 unit-scale `fp8_kv` 已通过质量门禁，也不把 scaled-FP8 的限定结果泛化为
  “所有 FP8 都质量无损”。
- 不声称已完成跨框架 page-table/allocator 全口径物理显存 Pareto；Prism 内部的
  process-NVML/KV bytes 已实测，但不能直接替代 vLLM/SGLang 的统一物理字节合同。
- 不声称 scaled-FP8 比 Prism BF16 更快，也不把 KV-limited sequence 上限写成
  online concurrency/goodput。
- 不把 offline output tok/s 当作 online serving goodput。
- 不把 P16/P17 的单进程重复媒体 Goodput 结果扩写为 unique-media、冷启动、跨进程/
  分布式网络缓存或通用线上服务全面超过 vLLM/SGLang；P17 已是内容寻址缓存，但当前
  fresh-object 数据仍显示 0--50% 重复率由 vLLM 领先。
- 不声称 phase-decomposed multimodal prefill 已保留或带来 online 加速；该原型已因
  loaded 退化删除。
- 不把 packed MLP 的小幅 decode TPOT 收益写成 online goodput或稳定 E2E 加速。
- 不声称已经验证 TP2、HTTP/gRPC、megakernel、PD 分离或投机解码。
- 不声称 NVFP4 或权重/激活量化已经实现、验证或优于 BF16。

## Acknowledgements

- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的轻量 engine 起点。
- vLLM / PagedAttention 的系统设计启发。
- FlashAttention 与 Triton 的高性能 attention/kernel 生态。

## License

MIT，见 [LICENSE](LICENSE)。
