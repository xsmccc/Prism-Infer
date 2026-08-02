# Prism-Infer

Prism-Infer 是一套面向 **Qwen3-VL** 的多模态推理引擎。本项目当前聚焦一个具体的
在线场景：用户上传一组图片后，围绕同一组视觉内容连续提出不同问题。

普通视觉缓存只能省去 Vision Encoder；语言模型仍需为每个新问题重算长视觉前缀。
Prism-Infer 将公共视觉上下文保存为物理压实的 Scaled-FP8 Prefix KV，在固定 KV
显存预算下驻留更多媒体，并让后续问题同时跳过 Vision/DeepStack 和公共语言 Prefill。

## 重复视觉上下文

请求使用带编号的 media-first 布局：

```text
Image 1: <image>
Image 2: <image>
...
Question: ...
```

运行时在 Scheduler admission 之前完成内容寻址查询。命中时直接挂接只读 Prefix
pages；未命中时才恢复或计算视觉特征、执行 Prefill、压实 KV 并写入缓存。缓存可以
使用整个暂时空闲的 KV Pool，活跃请求需要空间时先回收 tail clone，再淘汰完整条目。

![MuirBench working-set result](artifacts/working_set/performance/working_set_summary.png)

图中的三组 MuirBench 工作集分别包含 151、223、333 个 Dense Prefix pages，而固定
预算只有 220 pages。每组先建立一次媒体工作集，再运行 600 条 Zipf-1.0 多问题请求；
三套引擎使用相同图片、prompt token、到达顺序、greedy 参数和 4,282,122,240-byte
KV 预算。

在 `pressure` 工作集上：

| 引擎 | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | 进程峰值 | 重算 prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| **Prism Compact Prefix** | **102.401 ms** | **270.588 ms** | 350.512 ms | **774.318 ms** | 24,006 MiB | **83,018** |
| vLLM 0.25.1 | 131.483 ms | 581.976 ms | **323.228 ms** | 886.035 ms | **23,714 MiB** | 165,031 |
| SGLang 0.5.15.post1 | 494.911 ms | 1,623.782 ms | 855.944 ms | 2,270.308 ms | 24,284 MiB | 179,111 |

Prism 相对 vLLM 的 TTFT p50/p99 低 22.1%/53.5%，E2E p99 低 12.6%；E2E p50
慢 8.4%，进程峰值高 292 MiB。三者的输出吞吐均约为 64.1–64.2 tok/s，因为该负载
由固定到达率和 output 16 主导，不能据此声称吞吐领先。

Prism 内部对照显示：

- Compact 运行中的 Prefix pages 相对其 Dense-equivalent 从 4,412 减至 2,910
  （-34.0%）；
- 相对 Dense Prefix 路径，驻留媒体从 33 组增至 48 组（+45.5%）；
- 淘汰从 110 次减至 33 次，重算 prompt tokens 减少 56.8%；
- TTFT p50/p99 分别降低 17.4%/61.8%。

这组性能只适用于 RTX 5090、Qwen3-VL-8B、TP1、给定 KV 预算和重复多图提问场景。
它也不是等质量的通用引擎排名：在 49 条实际删除视觉 token 的 MuirBench 样本上，
Dense media-first 为 27/49，Uniform Compact 为 20/49。带编号的 media-first 在完整
85 条样本上为 46/85，官方交错布局为 49/85。项目同时公开容量、延迟和质量，不把有损
压实的性能收益描述成无代价领先；MVBench 的视频压实质量下降，因此视频删除默认关闭。

详细设计、质量对照、Trace 和限制见
[重复视觉上下文技术记录](docs/REPEATED_VISUAL_CONTEXT.md)。请求级 JSON、共用 plan
与派生表位于 [artifacts/working_set](artifacts/working_set/README.md)。

## 其他推理能力

- **Qwen3-VL 执行链路**：Vision Encoder、DeepStack、3D Position IDs、M-RoPE、
  Language Decoder 和 Sampling。
- **Scaled-FP8 Paged KV**：E4M3FN K/V 与 per-token、per-KV-head FP32 scale，贯通
  Store、Paged Attention、CoW、Swap、物理压实和 CUDA Graph Replay。
- **视觉 KV 物理压实**：移动保留 K/V 并释放真实 pages，同时保持原始 M-RoPE
  logical positions。图片主路径使用 query-agnostic Uniform；视频删除默认关闭。
- **torch.compile + CUDA Graph**：编译稳定的 QKV、QK-Norm 和 M-RoPE 子图，外层
  CUDA Graph 捕获完整 Decode，包括 Paged Attention、LM Head、选词和 TP2 NCCL。
- **在线与分布式**：Continuous Batching、Chunked Prefill、HTTP/SSE，以及单机
  双卡 Tensor Parallel。

### Decode 与 KV 容量

RTX 5090、TP1、batch 1、greedy output 128：

| 场景 | Prism TPOT | SGLang | vLLM |
|---|---:|---:|---:|
| 8 张 448×448 图片 | **9.8821 ms** | 10.3520 ms | 10.5276 ms |
| 16 帧 448×448 视频 | **9.8680 ms** | 10.3689 ms | 10.5278 ms |

Scaled-FP8 将同容量 KV 存储减少 48.44%；在约 4 GiB KV 预算下，token capacity 从
28,928 提升到 56,320（+94.69%）。双卡 TP2 单图 batch 1 的 TPOT 为 5.9701 ms，
较同配置 vLLM 的 6.1612 ms 低 3.10%，但 TTFT/E2E 仍慢于 vLLM，原因是 Vision
Encoder 在两个 rank 上重复执行。

## 快速开始

项目支持 Python 3.10–3.12。RTX 5090 实测环境使用 Python 3.12、PyTorch
2.11.0+cu130 和 Transformers 5.14.1。

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

启动 HTTP/SSE 服务：

```bash
prism-serve \
  --model "$PRISM_MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000
```

## 代码与文档

```text
prism_infer/
  engine/       Scheduler、Paged KV、Prefix Cache、Tensor Parallel
  models/       Qwen3-VL Language Model、Vision glue、DeepStack
  vision/       Vision Encoder、Attention、M-RoPE
  layers/       Linear、Norm、Attention、Sampler
  ops/          Triton KV Store、Paged Decode、Compaction、Fused Kernels
  serving/      HTTP/SSE Runtime
  analysis/     Benchmark 与 Profiler 分析
benchmarks/     Offline、Online、质量与 vLLM/SGLang 对比入口
configs/        Serving 和 TP2 配置
```

- [架构设计](docs/ARCHITECTURE.md)
- [完整结果](docs/RESULTS.md)
- [运行与复现](docs/REPRODUCIBILITY.md)

当前未实现 PP、多机 TP、MoE Expert Parallel、跨进程 Prefix Cache 和
OpenAI-compatible API。

## 致谢与许可

项目早期运行时结构参考了
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，之后扩展为面向 Qwen3-VL
的多模态推理实现。项目使用 [MIT License](LICENSE)。
