# 多模态 Prefill 与 Decode 交错执行

冷图片到达时，CPU Processor 已经可以在后台准备，但 GPU 上的 Vision 和语言 Prefill
仍会占住执行路径。此前的 HTTP 测量出现约 200 ms 的 token 停顿：正在生成的请求并不是
每个 Decode 都变慢了，而是在等另一个请求完成长 Prefill。

本轮沿用已有的 cooperative Prefill，在 Vision block 和语言 Transformer layer 之间
插入 Decode。它明显缩短最长停顿，但增加冷请求 TTFT，平均 TPOT 和 ITL p95 也没有改善。
因此保留为显式开启的延迟取舍，**没有改成全局默认，也不宣称提升了吞吐或超过 vLLM**。

## 实际改了什么

入口是 [LLMEngine._next_step_result](../prism_infer/engine/llm_engine.py)。当 Prefill
与 Decode 同时存在时，开启此配置会执行：

1. Scheduler 为 Prefill 分配页并建立 BatchPlan；`begin_prefill` 准备独立执行状态。
2. ModelRunner 执行若干 Vision blocks，或若干语言层，保存隐藏状态和下一层位置。
3. Scheduler 选择可以执行的驻留 Decode，请求走原有 CUDA Graph。
4. 恢复 Prefill 的 Attention Context，继续下一段；全部层完成后才采样、提交进度、发布 KV。

这不是把一个图片的 patches 随意拆开，也不是缩短 Attention 的可见上下文。每层仍处理
原来的 token 集合，层内 Attention 范围不变；没有修改 M-RoPE、量化方法或 Token Pruning。
`BatchPlan` 仍只有一个阶段，Prefill 和 Decode 是先后执行的不同批次，不是 mixed batch、
多 CUDA stream 同时运行，也不是跨卡 PD 分离。

原有 cooperative 策略会等待最多 250 ms 凑到 3 个 Prefill 请求；达到 3 个后，又改成整段
执行。普通 FCFS 路径现在不再等待凑批，也不再因 batch 达到 3 就放弃交错，直接采用配置
指定的层数。历史 `slo_aware` 策略保留原来的独立行为，本轮没有新增 SLO 或调整其参数。

### KV 空间不足时不能先抢占再丢掉操作

“队列里有 Decode”不等于“下一步 Decode 能分配到 KV”。例如总共 3 页：一个 Decoder
持有 1 页，暂停的 Prefill 持有 2 页，而 Decoder 的下一个 token 恰好要新页。此前会抢占
唯一 Decoder，随后因没有可执行请求而抛错；若抢占产生了 Swap 操作，也还没有机会执行。

现在 [Scheduler.schedule_resident_decode](../prism_infer/engine/scheduler.py) 只选择
能追加 token 的驻留请求，不抢占、不 Swap。缺页的请求暂留原队列；没有可选请求时先完成
当前 Prefill，再回到正常调度。这不保证在满池时也能维持短 ITL，只保证交错执行不会为了
腾空间破坏正在使用的请求状态。

### 取消暂停中的请求不能吞掉同批输出

旧的 `cancel_request()` 先强制完成整个 pending Prefill，却丢掉了返回的 StepResult。
取消 A 可能顺便计算 B 的首 token，但客户端收不到；B 若只生成一个 token，连完成事件也
会一起丢失。取消本身还会再次长时间阻塞 Decode。

现在取消 A 时丢弃整批未完成的 runner state，再取消 A；B 保留页表和已经提交的计算前沿，
下次重算尚未完成的 Prefill 片段。`begin/advance` 不会推进 Sequence 的计算前沿，也不会
提前发布这些未完成层的 KV。Scheduler 同时允许关闭 token chunking 时恢复这样的
PREFILLING 请求，避免 B 永久留在队列里。代价是同批存活请求可能重算部分工作，而不是维护
一套复杂的中间张量拆分和取消协议。

## 怎么测，测到了什么

复用上一轮原生 HTTP/SSE 客户端，不更改图片、Prompt 和生成参数。八张 448×448 纯色图片，
Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，RTX 5090，
TP1，PyTorch 2.11.0+cu130，驱动 580.142，Scaled-FP8 KV，64×256-token 页。
CPU 媒体缓存和 Prefix KV 开启，视觉输出缓存、token-level Chunked Prefill 关闭。

每个新服务进程先完成预热和同图换问题请求，再测独立长 Decode，以及三次“长请求输出至少
三个 token 后插入新颜色的八图请求”。长/短请求固定生成 128/16 tokens，greedy，
`ignore_eos=True`。HTTP 上传、服务端图片解码、准备和排队都计入客户端 TTFT/E2E，模型
加载和 Graph capture 不计入。纯色图片用于观察执行时序，不用于衡量真实任务准确率。

最终采用两个顺序相反的开关对照：9798 开 → 9799 关；9800 关 → 9801 开。每次先对三个
请求各自的指标取中位数，下表再取两个运行中位数的中位数；只有两个配对，不推断任意负载。
完整时间戳、SSE 事件与计算脚本见[实验目录](../artifacts/cooperative_prefill_20260908/README.md)。

| 指标 | 整段 Prefill | 每 8 层交错 | 含义 |
|---|---:|---:|---|
| 长请求最大 ITL | 225.18 ms | 42.19 ms | 最大停顿缩短约 81% |
| 长请求 ITL p95 | 12.36 ms | 28.27 ms | 更多间隔包含一个 Prefill 片段，反而上升 |
| 长请求平均 TPOT | 12.70 ms | 12.82 ms | 增加约 0.95%，没有 Decode 吞吐收益 |
| 长请求 E2E | 1,812.56 ms | 1,825.07 ms | 略有增加 |
| 注入冷请求 TTFT | 456.51 ms | 625.23 ms | 增加约 37% |
| 注入冷请求 E2E | 642.08 ms | 809.50 ms | 增加约 26% |

这里的“最大 ITL”和“ITL p95”均先在单条请求内统计，之后再汇总，不是整体 p99。
Prefill 的总计算量没有减少：原来一次长等待，现在分散成多次较短等待。Decode 被插入后，
冷请求到首 token 的时间相应变长。CPU/GPU 提交和工作集切换还会增加少量额外成本。
本轮未用独立测量拆分这部分成本，不把它全部归因于某一个 kernel。

最初直接开启旧配置（每层一次、保留凑批等待）时，冷 TTFT 达到 1,583.33 ms；去掉 FCFS
凑批并改为每 4 层时为 756.56 ms。之后只增加到每 8 层，减少交错次数，得到上表取舍。
这些探索结果完整保留，没有开展 scheduler 参数扫描，也没有把它称为全局最优参数。

## 输出差异与实现检查

最终四轮各完成 11 个请求、624 个输出 token，SSE 与最终 token 列表均相同。跨运行不是
严格逐 token 相等：以关闭交错的 9799 为参照，9798/9800/9801 分别有 2/1/1 条长请求的
第 128 个、也就是最后一个 token 不同，之前的输出相同；分歧前没有 EOS。关闭交错的
9799 与 9800 之间也有这类差异。因此不能将其直接归咎于交错，更不能宣称没有任何精度
损失。改变到达时序会改变 batching，但本轮没有进一步证明具体的数值误差来源。

每 4 层的探索运行另有两条长请求从第 122 个 token 起分歧，不混入最终配置的一致性统计。
固定输出长度保证没有通过少生成 token 换取延迟降低；这些 fixture 也不能给出 MuirBench
准确率变化。当前保留所有原始输出，不更新既有简历中的模型质量结论。

针对本轮改动的检查覆盖满池跨页 Decode、跳过缺页请求、取消同批成员及现有页生命周期和
Context 恢复。另一个真实 GPU 请求检查同时提交 3 个图片 Prefill，并分别在 Vision 和
语言阶段取消其中一个；存活请求的单 token 与原子执行一致，TOKEN/最终输出对应完整。
这验证请求和状态处理，不等于完整模型质量评估。

## 时间线怎么看

打开 [trace.json](../artifacts/cooperative_prefill_20260908/trace.json) 可看到 CPU NVTX
和 GPU kernel 两类轨道。单独的 Nsight job 9802 记录了 8 个 Vision quantum、5 个语言
quantum，12 个相邻 quantum 间隔中都出现 Decode Graph replay。原始
[Nsight 文件](../artifacts/cooperative_prefill_20260908/raw/prefill-9802.nsys-rep) 保留完整
CUDA 活动；带 profiler 的时延不参与上面的表格。

`EngineMetrics` 的 Prefill batch duration 从开始到提交计算，包含暂停期间插入的 Decode。
不能把它当作 GPU 净 Prefill 时间，也不能把这些重叠的 batch duration 相加当成 E2E。

## 使用与当前选择

完整配置见 [engine-cooperative.json](../artifacts/cooperative_prefill_20260908/engine-cooperative.json)。
关键参数为：

```json
{
  "enable_cooperative_prefill": true,
  "cooperative_prefill_layer_quantum": 8,
  "cooperative_prefill_vision_block_quantum": 8,
  "enable_chunked_prefill": false
}
```

该路径目前要求 TP1 和 CUDA Graph 后端，模型实现为 Qwen3-VL。全局开关仍默认 false，
全局 quantum 默认仍为 1；这里的 8 是上述工作负载使用的显式配置。现有多图 token
chunking 原子区间没有在本轮修改，视频也没有进行新的性能验证。

面试时应把它讲成一次有数据支持的调度取舍：先用时间线确认冷 Prefill 阻塞 Decode，再
通过层边界让出执行机会，修复取消与 KV 压力下的状态问题，最后发现最长停顿缩短，但
TTFT/平均 TPOT 没有同时改善。因此不把它包装成整体加速，也没有默认开启。
