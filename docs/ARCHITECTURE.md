# 架构设计

Prism-Infer 是一个面向 Qwen3-VL 的单机推理引擎。Tokenizer 和 Processor 来自
Hugging Face，模型 Forward、KV Cache、调度、Tensor Parallel 和 Serving Runtime
由项目自己实现。

## 1. 请求流程

```mermaid
sequenceDiagram
    participant Client
    participant Processor
    participant Cache as Multimodal Cache
    participant Scheduler
    participant Runner as Model Runner
    participant KV as Paged KV

    Client->>Processor: text + image/video
    Processor->>Processor: token IDs + M-RoPE positions
    Processor->>Cache: media hash + visual prompt
    alt cache hit
        Cache->>KV: acquire shared prefix pages
    else cache miss
        Processor->>Runner: vision inputs
        Runner->>Runner: Vision Encoder + DeepStack
        Runner->>KV: prefill and KV compaction
        KV->>Cache: store reusable prefix
    end
    Scheduler->>Runner: BatchPlan
    loop decode
        Runner->>Runner: compiled subgraph + CUDA Graph replay
        Runner->>KV: append K/V
    end
    Runner-->>Client: token stream
```

Scheduler 每一步生成一个 `BatchPlan`，其中包含请求阶段、token 数、页表和调度操作。
Executor 按顺序完成页复制、Swap、Prefill、Decode 和资源回收。请求可以处于 waiting、
running、swapped、completed、cancelled 或 failed 状态。

## 2. Qwen3-VL 模型

模型路径包括：

- Text Embedding、Decoder、LM Head；
- Q/K RMSNorm、Grouped-Query Attention 和 MLP；
- Vision Transformer、Patch Merger 和 DeepStack；
- 单图、多图、视频以及混合 batch；
- Qwen3-VL 3D Position IDs 和 M-RoPE delta；
- Greedy 和 Temperature Sampling。

视觉 KV 压实时，序列同时保存逻辑 token 位置和物理 KV 位置。删除一部分视觉 KV
只会改变页表与 Attention 读取位置，不会改变保留 token 的 M-RoPE 坐标。

## 3. torch.compile 与 CUDA Graph

Decode 中并不是所有内容都适合交给 compiler。Prism 将稳定计算和动态状态分开：

- `torch.compile` 处理 QKV Projection、QK-Norm 和 M-RoPE；
- CUDA Graph 捕获固定 batch bucket 的完整 GPU Decode；
- Paged KV 页表、context length、slot mapping 和 FP8 scale 使用固定地址 Tensor；
- Prefill、请求加入、缓存淘汰和页分配仍在普通运行时中执行。

TP1 的 Graph 可以覆盖模型 Forward、LM Head 和 greedy token selection。TP2 中，
每个 rank 运行自己的 compiled QKV 子图，外层 Graph 再捕获 Paged Attention、NCCL
AllReduce、LM Head 和分布式 top-1。

```text
host state update
  -> rank-local compiled QKV/QK-Norm/M-RoPE
  -> KV store
  -> Paged Attention
  -> NCCL AllReduce
  -> LM Head
  -> distributed greedy top-1
```

低精度候选选择不会直接决定输出。最终 token 由 FP32 重排得到，候选间距过小时回到
完整精度路径。

## 4. Tensor Parallel

TP2 切分语言模型，Vision Encoder 暂时复制执行。

Column-parallel：

- Q/K/V Projection；
- MLP Gate/Up；
- Vocabulary Embedding；
- LM Head；
- 每张卡对应的 KV heads。

Row-parallel：

- Attention Output Projection；
- MLP Down Projection；
- 使用 NCCL AllReduce 合并部分结果。

Greedy Decode 时，每个 rank 先计算本地最大 logit 和全局 token ID，然后通过一次小型
AllGather 决定最终 token，不需要汇总完整词表。非 greedy sampling 仍会收集完整
logits。

batch1 快路径只发送当前 token、M-RoPE position、KV slot、context length 和 block
table。每个 rank 将这些数据写入自己的 pinned host buffer，然后 replay CUDA Graph。
其他 batch size 继续使用通用 `BatchPlan` 路径。

## 5. Paged KV 与 Scaled-FP8

每条序列通过 block table 映射到物理 KV pages。Prefix Sharing 使用只读共享页；写入
共享尾页时执行 Copy-on-Write。Swap 会同时移动 payload、scale 和页元数据。

Scaled-FP8 格式：

```text
K/V payload: E4M3FN
K scale:     FP32[token, kv_head]
V scale:     FP32[token, kv_head]
```

scale 与 K/V 一起经历 Store、Paged Attention、CoW、Swap、Compaction 和 Graph
Replay。直接将 BF16 KV 转成 unit-scale FP8 的质量较差，因此最终实现使用动态
per-token、per-head scale。

## 6. 视觉 KV 压实

Prefill 后，Coordinator 按选定策略产生视觉 token 保留表。运行时支持 Uniform 和
Attention 两种选择方式；重复提问主路径使用 query-agnostic Uniform，因为它不依赖
某一道问题，可以让同一份压实 KV 被后续问题直接复用。Attention Top-k 保留在质量
对照中，用来测量“每题重算选择”和“沿用第一题选择”的差异。

得到保留表后，运行时：

1. 计算要保留的视觉 token；
2. 将对应 K/V 和 scale 移动到连续位置；
3. 更新 block table 和 physical context length；
4. 释放已经空出的 pages；
5. 保留原始 M-RoPE logical positions。

与 Attention Mask 不同，物理压实会真正释放 KV pages，让后续请求可以使用这些空间。
主配置为图片 `keep_ratio=0.6`、最少保留 768 个视觉 token；低于该数量的单图不会被
压实。页数收益还受到 256-token page 粒度影响，因此不能简单按 40% 估算。视频默认
不删除 token；只有显式提供 `visual_pruning_video_min_keep_tokens` 时才启用视频压实。

## 7. 多模态前缀缓存

缓存分为两层：

```text
model / processor version + media bytes + dtype + shape
    -> Processor、Vision、DeepStack cache

上面的 key + visual prompt tokens
    -> compacted prefix KV cache
```

重复提问路径把有序媒体放在问题之前，并保留显式编号：

```text
Image 1: <image>
Image 2: <image>
...
Question: ...
```

编号用于保持多图对应关系；最后一个视觉占位符之前的精确 token 序列是可复用公共
前缀，问题文本及其后续生成 token 不进入缓存。

请求进入 Scheduler 之前先计算媒体与公共 prompt 的身份并查询 Prefix Cache：

```text
processor tokens + media identity
  -> direct prefix lookup
  -> hit: skip Vision/DeepStack hydration, attach shared KV pages
  -> miss: hydrate/build Vision/DeepStack, normal prefill, compact and retain pages
  -> scheduler admission
```

Prefix ID 由模型与 Processor 布局、按顺序排列的媒体 SHA256、公共 prompt token 数和
完整 token SHA256 直接得到，字典查找为 O(1)。命中后仍比较媒体 key、公共长度和完整
token 序列；摘要碰撞会直接报错，不会复用错误 KV。文件输入按内容计算身份，不依赖
路径或 Python 对象地址。

早期查询和真正分配 KV 之间可能发生淘汰。因此原始媒体 Tensor 保留到分配完成；若条目
已经被回收，请求自然回到完整 Vision + Prefill 路径。运行时分别记录
`pre_admission_hits`、`visual_hydration_skips` 和 `stale_probe_fallbacks`，可以确认命中
是否真的绕过视觉计算。

Prefix Cache 持有只读完整页。最后一页未填满时，请求获得自己的 tail page；tail page
结束使用后回到复用池，避免每次重新申请和复制。缓存不再只占 KV Pool 的八分之一，
而是可以使用全部暂时空闲页。活跃请求分配、追加、CoW 或 Swap-in 需要空间时，先回收
空闲 tail page，再按 `benefit_tokens × (1 + hits) / resident_pages` 淘汰完整条目；仍被
活跃请求引用的共享页不会释放。

## 8. 调度与 Serving

运行时记录 TTFT、TPOT、E2E、吞吐、KV pages 和 Cache 命中。Scheduler 支持
Continuous Batching、FCFS 和 Chunked Prefill。

连续加入较重的 Vision Prefill 会打断已有 Decode，因此调度器会控制 Prefill 粒度。
HTTP Runtime 支持普通 JSON 响应、SSE Token Stream、取消请求和退出时释放显存。

## 9. 当前实现情况

- TP1 和 TP2 已在 RTX 5090 上完成图像、视频、混合 batch 和 HTTP/SSE 测试；
- TP2 的 Vision Encoder 仍然复制执行，尚未实现 Vision Parallel；
- Dynamic Vision Tensor Graph 在混合 shape 下会改变首 token，因此默认关闭；
- Prefix Cache 位于单个 Engine Process 内，尚未做跨进程或跨机器共享；
- 当前 Serving API 为项目自有格式，不兼容 OpenAI API。
