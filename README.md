# Prism-Infer

Prism-Infer 是面向 Qwen3-VL 的多模态推理引擎，支持图片、视频输入和流式服务。
项目重点是同一组图片被多个独立请求反复查询时的 Prefix KV 复用，以及固定显存预算下的
FP8 KV 存储。它不是对 vLLM 或 SGLang 的通用替代。

## 多模态 Prefix Cache

将视觉输入置于问题前，形成跨请求一致的公共前缀。只有 Processor/Encoder Cache 时，
可以复用预处理或视觉编码结果，但不能直接省掉语言模型的公共前缀 Prefill；Prefix KV
Cache 则保留各 Transformer 层已经计算的 K/V。这里讨论的是独立请求之间的缓存复用，
不是已有会话 KV 驻留时又把同一段上下文重新计算一遍。

请求入队前按媒体内容、Processor 布局和公共 token 前缀查询缓存。命中覆盖完整视觉输入
时，跳过 Vision Encoder、DeepStack 和公共前缀 Prefill；部分命中只复用连续一致的前缀，
继续计算未共享的图片和文本。图片内容相同但左侧上下文不同，不会独立复用其语言模型 KV。

页管理遵循三个规则：完整页的 KV 计算完成后才加入命中索引；共享尾页只复制公共前缀行，
不继承旧问题的哈希；空闲缓存可借用整个 KV Pool，但活跃请求引用的页不能回收。多个
缓存条目共享同一页时，按唯一物理页和实际引用计算可回收容量。

## 推理实现

- 适配 Qwen3-VL Vision Encoder、DeepStack、M-RoPE 和 Language Decoder；支持
  Continuous Batching、Chunked Prefill、HTTP/SSE 和单机 TP2。
- FP8 Paged KV Cache：E4M3FN K/V，为每个 token、每个 KV head 分别保存 K/V 的 FP32
  scale；贯通 KV 写入、Paged Attention、Prefix 共享、CoW、Swap 和页回收。
- Prefix 命中的 Scaled-FP8 Prefill 从调度元数据读取长度，批量 gather/反量化 KV，再用
  右下对齐的因果 SDPA 处理问题后缀；支持的 CUDA 后端直接执行 GQA，不手工复制 K/V heads。
- Decode 按 batch bucket 捕获 CUDA Graph。TP1 的 `torch.compile` 路径编译 Attention
  输出投影和 FP8 LM-head 候选投影，候选再用原始权重进行 FP32 重排。
- TP2 支持按完整图片分配 Vision Encoder 工作，一次收集主特征和全部 DeepStack，
  恢复原始媒体顺序后进入语言模型。完整视觉 Prefix 命中时不再向 worker 发送图片 Tensor。
- 普通生成、Online 和 HTTP 入口共用 CPU 媒体预处理缓存。HTTP 媒体准备在后台线程完成，
  模型与 KV 状态仍由同一个 owner 线程管理，减少冷图片到达对已有 Decode 的阻塞。

2026-09-07 修复了提前发布未计算 KV、尾页旧哈希、共享缓存页回收少算和 FP8 压实地址
溢出。对应复现、GPU 检查和执行路径记录见[修复说明](docs/RUNTIME_FIXES_20260907.md)。
本轮没有重跑三引擎端到端排名。

2026-09-08 增加了[多图 Encoder 数据并行](docs/MULTI_GPU.md)，并修复 TP2 Prefix
尾页复制的命令分发和图片布局字段恢复。两张 5090、八图视觉阶段从 59.33 ms 降至
31.81 ms（含特征聚合）；这不是端到端 1.86×。同资源的 TP2、双 TP1 副本和 PP2
方向比较、数值差异及请求级结果均在该文档中说明。

随后统一了[共享预处理与 HTTP 请求路径](docs/SHARED_PREPROCESSING.md)。在同一组八图
HTTP 请求中，同图换问题 TTFT 中位数由 317.96 降至 189.14 ms；Decode 中插入冷图片时，
每个长请求的最大 token 间隔的中位数由 388.49 降至 216.54 ms。冷图片请求自身的 TTFT
略有上升，完整原始结果和 CPU/GPU 重叠 Trace 均保留；这不是竞品排名或总体 p99 结果。

## 结果与适用范围

Qwen3-VL-8B、RTX 5090 上的 KV 容量记录如下，scale 开销已计入：

| 配置 | Token capacity | KV 存储 |
|---|---:|---:|
| BF16 | 28,928 | 4,068.000 MiB |
| Scaled-FP8，同容量 | 28,928 | 2,097.562 MiB |
| Scaled-FP8，约 4 GiB | 56,320 | 4,083.750 MiB |

同 token capacity 下存储减少 **48.44%**；约 4 GiB 预算下容量增加 **94.69%**。
这两个数字描述存储，不代表量化在数学上无损。

独立的历史 batch-1 Decode 测量中，八图/视频 TPOT 为 9.8821/9.8680 ms，较当时
SGLang 低 4.54%–4.83%、较 vLLM 低 6.13%–6.27%。这些不是本轮 APC 修复后的高并发
结果，也不能代替 Prefill、TTFT 或吞吐评价。完整配置与历史结果见[Results](docs/RESULTS.md)。

视觉 Token Pruning 是可选研究路径，默认保留全部视觉 token。早期剪枝质量和 Compact
延迟记录受 FP8 压实地址溢出影响，不能继续作为算法取舍或跨引擎领先依据。修复后的
85 题 MuirBench 记录为 Dense 46/85、Uniform 47/85；实际删除的 49 题为 27/49、28/49，
仅说明该样本未观察到准确率下降。原始记录见[压实修复后的实验](artifacts/cache_pressure_20260813/README.md)。

## 快速开始

RTX 5090 开发环境使用 Python 3.12、PyTorch 2.11.0+cu130 和 Transformers 5.14.1。

```bash
git clone https://github.com/xsmccc/Prism-Infer.git
cd Prism-Infer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[blackwell,serving]"

export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
python example.py
prism-serve --model "$PRISM_MODEL_PATH" --host 127.0.0.1 --port 8000
```

`compression_mode` 默认是 `off`；FP8 KV 和视觉 Token Pruning 需要显式选择相应配置。
运行参数见[Reproducibility](docs/REPRODUCIBILITY.md)。

## 文档与代码

- [Architecture](docs/ARCHITECTURE.md)：模型适配、KV 布局、缓存与调度。
- [重复视觉上下文](docs/REPEATED_VISUAL_CONTEXT.md)：请求路径、历史实验及结果解释。
- [Results](docs/RESULTS.md)：区分存储、Decode、在线工作集和质量测量。
- [本轮修复与执行证据](docs/RUNTIME_FIXES_20260907.md)。
- [多卡多模态实现与取舍](docs/MULTI_GPU.md)：Encoder DP、TP2 Prefix、双副本和 PP2 参照。
- [共享预处理与后台准备](docs/SHARED_PREPROCESSING.md)：统一缓存、线程所有权、取消与 HTTP 实测。
- [历史请求级 JSON 与 Trace](artifacts/working_set/README.md)。
- [相关工作](docs/RELATED_WORK.md)、[未采用方案与历史实验](docs/REJECTED_EXPERIMENTS.md)。

主要实现位于 `prism_infer/engine`、`models`、`vision`、`ops` 和 `serving`。
Prefix Cache 位于单个 Engine Process 内；当前未实现 PP、多机 TP、MoE Expert Parallel
或 OpenAI-compatible API。Scaled-FP8 Prefill 仍需要将 paged KV gather 到临时连续张量，
并非直接读取量化页的融合 Prefill kernel。

## 致谢与许可

项目早期运行时结构参考了
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，随后扩展为 Qwen3-VL 多模态推理实现。
项目使用 [MIT License](LICENSE)。
