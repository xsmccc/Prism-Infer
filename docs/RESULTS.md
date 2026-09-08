# 性能与质量结果

## 历史模型实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32 GB |
| Driver / CUDA | 580.105.08 / 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.14.1 |
| Model | Qwen3-VL-8B-Instruct |
| Model revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| External engines | vLLM 0.25.1 / SGLang 0.5.15.post1 |

工作集结果、batch-1 Decode、KV 容量和 TP2 使用不同的固定协议。下面分别陈述，不把
不同协议的数字拼成一次实验。请求级记录和派生表位于
[`artifacts/working_set`](../artifacts/working_set/README.md)。

2026-09-07 的代码修复和 GPU 检查使用单独记录，不把历史模型结果视作本轮复测：
[运行时修复说明](RUNTIME_FIXES_20260907.md)。

## 1. 重复视觉上下文

### 1.1 工作集与比较协议

性能工作集只包含“同一有序媒体至少对应两个不同问题”的 MuirBench 媒体组。每组先
冷建立一次前缀，随后运行 600 条 Zipf-1.0 请求；Poisson 4 req/s、greedy、output 16。
每条测量请求都切换到该媒体组的另一个问题。

| Workset | 媒体组 | 不同问题 | Dense Prefix pages | 220-page 预算占比 |
|---|---:|---:|---:|---:|
| fit | 21 | 42 | 154 | 70.0% |
| knee | 28 | 56 | 224 | 101.8% |
| pressure | 42 | 85 | 312 | 141.8% |

三套引擎使用完全相同的图片、media-first prompt、post-tokenization token IDs、请求顺序、
到达时间和生成参数。KV 字节预算固定为 4,282,122,240 bytes。Prism 使用 Scaled-FP8
Paged KV 和 256 MiB Vision Cache；vLLM 使用 FP8 KV、Automatic Prefix Caching 和
1 GiB Processor Cache；SGLang 使用 FP8 KV、Radix Cache 与 `mm_global_cache`。进程峰值
来自 NVML compute-process 采样，不用整卡 `memory.used` 代替。

### 1.2 压实地址修复后的历史对照

下面的数据来自 2026-08-13 的 `c4c8550` 加 int64 压实修复和当时的 benchmark 改动，
不是 2026-09-07 APC 修复后的 main 测量。原始请求记录已补入
[`artifacts/cache_pressure_20260813`](../artifacts/cache_pressure_20260813/README.md)。

相同 220-page 预算、600 请求下的 Prism 内部对照：

| 路径 | 缓存驻留媒体组 | Prefix 淘汰 | TTFT p50 / p99 | E2E p50 / p99 |
|---|---:|---:|---:|---:|
| Dense Prefix | 27 | 96 | 101.356 / 721.983 ms | 379.096 / 1,062.074 ms |
| Uniform Compact Prefix | 40 | 15 | 90.264 / 523.621 ms | 324.390 / 845.583 ms |

TTFT p50/p99 分别降低 10.94%/27.47%。计时起点是 controller 开始处理到期请求、尚未执行
框架相关 Prompt/media 准备的时刻；初始化、Graph capture 和 42 条 population 请求均已完成。

3,941 → 2,762（-29.92%）来自命中请求携带的冷构建压实记录累计，并非某一时刻的页池
驻留量。不要把它写成“整个 KV Pool 的实际页数减少 29.92%”。

### 1.3 三引擎的容量敏感性

每个引擎分别运行三次 352/220-page 对应字节预算的配对实验：两次大容量到小容量，一次
反序。先计算每次配对中小容量相对大容量的变化，再取三次变化的中位数。

| 引擎 | 首 token p50 增幅中位数 | 三次范围 | 首 token 均值增幅中位数 |
|---|---:|---:|---:|
| Prism | +1.63% | +0.78%–+1.67% | -0.02% |
| vLLM 0.25.1 | +14.30% | +6.37%–+14.93% | +35.34% |
| SGLang 0.5.15.post1 | +36.98% | +23.97%–+41.96% | +37.68% |

这组历史记录显示所测 Prism 配置对容量收缩较不敏感，不是三套引擎绝对延迟的通用排名。
计时都不包含启动，但框架原生 API 和预处理路径并不相同；跨引擎完整输出也不完全一致。
由 `prompt_tokens - cached_tokens` 得到的数值是逻辑缓存覆盖代理量，不是相同定义的
Vision 重算或 GPU Prefill 工作量。固定 4 req/s、output 16 也使吞吐主要受请求到达率限制。

旧页面引用的 Compact TTFT 101.692/497.899 ms 和“较 vLLM 降低 24.52%/29.85%”来自
压实修复前记录，不再作为有效性能结论。原始文件保留用于解释历史问题。

### 1.4 质量记录的更正

FP8 压实 kernel 曾以 int32 计算跨 K/V 层偏移，在实际 KV Pool 大小时溢出。因此旧
MuirBench 27/49 → 20/49、MVBench 183/252 → 113/252 混入了实现错误，不能把差异
直接归因于视觉 Token Pruning。

修复后的 MuirBench 配对记录：

| 样本 | Dense Scaled-FP8 | Uniform Compact |
|---|---:|---:|
| 全部 85 题，media-first | 46/85 | 47/85 |
| 实际删除视觉 token 的同一组 49 题 | 27/49 | 28/49 |

两种配置有 3/85 个答案不同。这支持“该样本未观察到准确率下降”，不证明等价或准确率提升。
这是 Dense FP8 与 FP8 加剪枝的对照，**不是 BF16 与 FP8 量化损失测试**。

Dense 官方交错布局与 media-first 的历史准确率为 49/85 和 46/85；这反映 Prompt 布局
也会影响答案。Attention Top-k、MVBench Compact 尚无修复后重跑结果，不能继续据旧数据
比较 selector。DocVQA 的 190 条样本未触发 token 删除，不能用于证明剪枝保留 OCR 精度。

默认仍不删除视觉 token。FP8 量化和 Token Pruning 都不应表述为数学上无损；前者改变数值
表示，后者改变可见上下文。更多说明见[未采用方案与历史实验](REJECTED_EXPERIMENTS.md)。

### 1.5 Prefix-hit Trace

历史 Nsight Systems capture 分别包含冷请求和同媒体不同问题的 Prefix 命中请求：

- 冷请求出现一次 `prism::model.vision.embedding_cache_miss`；
- Prefix-hit range 没有 Vision/DeepStack range；
- `visual_hydration_skips` 增加 1，`stale_probe_fallbacks` 为 0；
- 命中请求复用 145/275 个 prompt tokens；
- cold/hit GPU busy time 为 44.365/19.576 ms。

Trace 用于确认执行路径，不替代上面的非 profiler 延迟测量。机器可读观察项位于
[`trace_audit.json`](../artifacts/working_set/trace/trace_audit.json)。

## 2. TP1 Decode

协议：batch 1、greedy、output 128、warmup 2、repeat 5；三套引擎使用相同的
Qwen3-VL prompt tokens。

| 场景 | 引擎 | TPOT | TTFT | E2E |
|---|---|---:|---:|---:|
| 8 张 448×448 图片 | **Prism** | **9.8821 ms** | **245.349 ms** | **1,598.843 ms** |
| | SGLang | 10.3520 ms | 284.844 ms | 1,600.005 ms |
| | vLLM | 10.5276 ms | 290.574 ms | 1,628.751 ms |
| 16 帧 448×448 视频 | **Prism** | **9.8680 ms** | **240.175 ms** | **1,601.801 ms** |
| | SGLang | 10.3689 ms | 390.149 ms | 1,707.185 ms |
| | vLLM | 10.5278 ms | 323.819 ms | 1,673.800 ms |

Prism TPOT 比 SGLang 低 4.54%–4.83%，比 vLLM 低 6.13%–6.27%。该结果只覆盖 RTX
5090、TP1、batch 1 的固定 Decode 路径，不代表高并发吞吐排名。

## 3. KV Cache 容量

| 配置 | Pages / capacity | KV bytes | NVML peak | Torch peak |
|---|---:|---:|---:|---:|
| BF16 | 113 / 28,928 tokens | 4,068.000 MiB | 23,938 MiB | 21,637.368 MiB |
| Scaled-FP8，同容量 | 113 / 28,928 tokens | 2,097.562 MiB | 21,966 MiB | 19,667.298 MiB |
| Scaled-FP8，约 4 GiB | 220 / 56,320 tokens | 4,083.750 MiB | 23,952 MiB | 21,653.298 MiB |

Scaled-FP8 在同 token capacity 下将 KV 存储减少 48.44%，进程显存峰值减少 1,972 MiB
（8.24%）；相同约 4 GiB KV 预算下，capacity 从 28,928 增至 56,320 tokens
（+94.69%）。K/V 使用 E4M3FN payload 和 per-token、per-KV-head FP32 scale；scale
开销包含在上述容量中。

## 4. 双卡 TP2

协议：单张 448×448 图片、210 prompt tokens、batch 1、greedy、output 32、warmup 1、
repeat 3、`max_model_len=512`。

| 引擎 | TPOT | TTFT | E2E | 输出关系 |
|---|---:|---:|---:|---|
| **Prism** | **5.9701 ms** | 86.057 ms | 281.098 ms | 与 vLLM 的 32 个 token 相同 |
| vLLM | 6.1612 ms | 61.251 ms | 252.744 ms | 参考输出 |
| SGLang | 5.9701 ms | 45.452 ms | 230.601 ms | 从第 21 个 token 起不同 |

Prism TPOT 比 vLLM 低 3.10%，但 TTFT/E2E 更慢。Vision Encoder 仍在两个 rank 上重复
执行，测量主机也没有直接 GPU P2P/NVLink。Nsight 归因中 cuBLAS BF16 GEMV 约占 84%
GPU 时间，NCCL AllReduce 约 8%，Paged Attention 约 2.6%；结论是语言 Decode 切分
有效，但多模态端到端仍受复制 Vision 路径限制。

## 5. 适用范围

- 主工作集结果只覆盖 Qwen3-VL-8B、RTX 5090、TP1、固定 KV 字节预算和重复多图提问；
- 图片压实只有有限的修复后质量样本，默认不删除视觉 token；视频删除也默认关闭；
- Prefix Cache 位于单个 Engine Process 内，不跨进程或机器共享；
- 图片级 Vision Encoder 数据并行已实现，见[多卡记录](MULTI_GPU.md)；PP、MoE Expert
  Parallel 和多机推理尚未实现；
- Serving API 为项目自有格式，不兼容 OpenAI API。

## 6. 2026-09-08 HTTP Prefill 交错执行

TP1、Scaled-FP8 KV、八图 fixture，两组反序开关对照中，每 8 层插入 Decode 将单请求
最大 ITL 的汇总值从 225.18 降至 42.19 ms；平均 TPOT 从 12.70 增至 12.82 ms，ITL p95
从 12.36 增至 28.27 ms，冷请求 TTFT 从 456.51 增至 625.23 ms。因此没有默认开启。
这些 ITL 值先在每个请求内计算，再汇总，不是整体 p99。

本轮是执行时序取舍，不是 vLLM 排名或质量测试；跨运行有少数长请求最后一个 token
不同，关闭交错的两次运行之间也有差异。配置、原始 JSON、取消修复和 Nsight 时间线见
[PREFILL_INTERLEAVING.md](PREFILL_INTERLEAVING.md)。
