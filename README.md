# Prism-Infer

Prism-Infer 是面向 Qwen3-VL 的压缩感知多模态推理引擎。项目聚焦一个具体的在线场景：
用户上传一组图片后，围绕相同视觉内容连续提出不同问题。

普通 Processor 或 Vision Cache 只能省去视觉编码，语言模型仍要为每个问题重新 Prefill
长视觉前缀。Prism-Infer 将问题无关的视觉上下文保存为物理压实的 Scaled-FP8 Paged
Prefix KV；后续问题命中时直接挂接共享页，同时跳过 Vision Encoder、DeepStack 和公共
语言 Prefill。在固定 KV 显存预算下，压实后的前缀可以扩大可驻留媒体工作集，降低容量
压力下的淘汰和重算。

## 重复视觉上下文

请求使用带编号的 media-first 布局：

```text
Image 1: <image>
Image 2: <image>
...
Question: ...
```

缓存身份由模型与 Processor 配置、有序媒体内容 SHA256、公共前缀长度和完整 token
SHA256 共同构成。请求在 Scheduler admission 前直接查询 Prefix Cache；命中后挂接只读
页，未命中才恢复或计算视觉特征并执行 Prefill。缓存可以使用整个暂时空闲的 KV Pool，
活跃请求需要空间时先回收空闲尾页，再淘汰完整 Prefix Entry。

![MuirBench working-set result](artifacts/working_set/performance/working_set_summary.png)

图中的 `fit`、`knee`、`pressure` 分别包含 21、28、42 个重复媒体组和 42、56、85 个
不同问题。每组先建立一次媒体前缀，再运行 600 条 Zipf-1.0 请求；600 条测量请求都切换
到该媒体组的另一个问题，不是重复完全相同的 prompt。Prism、vLLM 和 SGLang 使用相同
图片、prompt token、请求顺序、到达时间、生成参数和 4,282,122,240-byte KV 预算。

### 容量压力下的三引擎结果

`pressure` 工作集需要 312 个 Dense Prefix pages，而可用预算为 220 pages：

| 引擎 | TTFT p50 / p99 | E2E p50 / p99 | 进程显存峰值 | 重算 prompt tokens |
|---|---:|---:|---:|---:|
| **Prism Compact Prefix** | **101.692 / 497.899 ms** | 334.834 / **832.635 ms** | **24,002 MiB** | **75,951** |
| vLLM 0.25.1 | 134.719 / 709.764 ms | **325.141** / 1,026.414 ms | 24,440 MiB | 165,678 |
| SGLang 0.5.15.post1 | 305.770 / 1,165.342 ms | 523.622 / 1,909.157 ms | 26,598 MiB | 181,294 |

在该场景下，Prism 相对 vLLM 的 TTFT p50/p99 分别低 24.52%/29.85%，E2E p99 低
18.88%，重算 prompt tokens 少 54.16%，进程峰值低 438 MiB；E2E p50 仍慢 2.98%。
固定 4 req/s、output 16 的请求流把三套引擎的输出吞吐限制在约 64.2 tok/s，因此这里不把
吞吐写成领先指标。`fit` 和 `knee` 上 Prism 的 TTFT p50 较低，但 tail latency 与 E2E
仍落后 vLLM，优势只在工作集超过 KV 容量后形成。

### 为什么压实会改善工作集

同一 `pressure` 请求流上的 Prism 内部对照：

| 路径 | Prefix 实际页 / Dense 等价页 | 驻留媒体 | Prefix 淘汰 | 重算 tokens | TTFT p50 / p99 |
|---|---:|---:|---:|---:|---:|
| Dense Scaled-FP8 Prefix | 3,413 / 3,413 | 27 | 96 | 188,169 | 124.994 / 695.924 ms |
| **Compact Scaled-FP8 Prefix** | **2,762 / 3,941** | **40** | **15** | **75,951** | **101.692 / 497.899 ms** |

Compact 路径将运行中 Prefix 页数减少 29.92%，可驻留媒体增加 48.15%，淘汰减少
84.38%，重算 token 减少 59.64%；对应 TTFT p50/p99 分别降低 18.64%/28.46%。

压实并非无损。MuirBench 的 85 个跨问题样本中，官方交错 Prompt 的 Dense 结果为
49/85，media-first Dense 为 46/85；在确实删除了视觉 token 的 49 个样本上，Dense 为
27/49，Uniform Compact 为 20/49。DocVQA 样本受 768-token 最低保留量限制，没有发生
删除；MVBench 视频压实从 183/252 降至 113/252，因此视频 token 删除默认关闭。

完整工作集、质量对照和 Trace 见
[重复视觉上下文技术记录](docs/REPEATED_VISUAL_CONTEXT.md)。

## 推理引擎能力

- Qwen3-VL Vision Encoder、DeepStack、3D Position IDs、M-RoPE、Language Decoder 与
  Sampling；支持单图、多图、视频和混合 batch。
- Scaled-FP8 Paged KV：E4M3FN K/V 与 per-token、per-KV-head FP32 scale，贯通 KV
  Store、Paged Attention、Copy-on-Write、Swap、物理压实和 CUDA Graph Replay。
- `torch.compile` 编译稳定的 QKV、QK-Norm、M-RoPE 等无状态 Decode 子图；外层 CUDA
  Graph 按 batch bucket 捕获完整 GPU Decode。
- Continuous Batching、Chunked Prefill、HTTP/SSE Serving，以及单机双卡 Tensor
  Parallel。

### Decode、KV 容量与 TP2

以下结果属于独立协议，不与上面的在线工作集混为一次实验：

| 测量 | Prism | 对照或变化 |
|---|---:|---:|
| TP1，8 张 448×448 图片，batch 1 TPOT | **9.8821 ms** | SGLang 10.3520 ms；vLLM 10.5276 ms |
| TP1，16 帧 448×448 视频，batch 1 TPOT | **9.8680 ms** | SGLang 10.3689 ms；vLLM 10.5278 ms |
| Scaled-FP8，同 token capacity 的 KV 存储 | **-48.44%** | 进程显存峰值 -8.24% |
| 约 4 GiB KV 预算的 token capacity | **56,320** | BF16 28,928；+94.69% |
| TP2，单图 batch 1 TPOT | **5.9701 ms** | vLLM 6.1612 ms；-3.10% |

TP2 的 TTFT/E2E 仍慢于 vLLM，因为 Vision Encoder 在两个 rank 上重复执行；项目没有
把局部 Decode TPOT 结果描述成端到端领先。详细协议见[结果汇总](docs/RESULTS.md)。

## 快速开始

RTX 5090 实测环境使用 Python 3.12、PyTorch 2.11.0+cu130 和 Transformers 5.14.1。

```bash
git clone https://github.com/xsmccc/Prism-Infer.git
cd Prism-Infer

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[blackwell,serving]"

export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
python example.py
```

HTTP/SSE 服务：

```bash
prism-serve --model "$PRISM_MODEL_PATH" --host 127.0.0.1 --port 8000
```

## 代码与证据

```text
prism_infer/
  engine/       Scheduler、Paged KV、Prefix Cache、Tensor Parallel
  models/       Qwen3-VL Language Model、Vision glue、DeepStack
  vision/       Vision Encoder、Attention、M-RoPE
  layers/       Linear、Norm、Attention、Sampler
  ops/          Triton KV Store、Paged Decode、Compaction、Fused Kernels
  serving/      HTTP/SSE Runtime
  analysis/     Benchmark 与 Profiler 分析
benchmarks/     Offline、Online、质量与三引擎工作集入口
configs/        Serving 与 TP2 配置
```

- [架构设计](docs/ARCHITECTURE.md)
- [性能与质量结果](docs/RESULTS.md)
- [运行与复现](docs/REPRODUCIBILITY.md)
- [相关工作与项目边界](docs/RELATED_WORK.md)
- [请求级 JSON、图表与 Trace](artifacts/working_set/README.md)

适合直接查看或截图的机器可读摘要为
[`artifacts/working_set/highlights.json`](artifacts/working_set/highlights.json)；Prefix 命中路径见
[`trace_audit.json`](artifacts/working_set/trace/trace_audit.json)。

当前未实现 PP、多机 TP、MoE Expert Parallel、跨进程 Prefix Cache 和 OpenAI-compatible
API。项目结果限定于文档记录的 Qwen3-VL-8B、RTX 5090、输入、batch、KV 预算和软件
版本，不主张通用场景全面优于 vLLM 或 SGLang。

## 致谢与许可

项目早期运行时结构参考了
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，随后扩展为面向 Qwen3-VL 的
多模态推理实现。项目使用 [MIT License](LICENSE)。
