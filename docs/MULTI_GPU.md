# 多卡多模态：Encoder DP、语言 TP 与副本吞吐

这轮工作针对一个具体问题：Prism 的语言模型已经做 TP2，但此前每个 rank 都完整计算同一批图片的 Vision Encoder。现在可以让 TP ranks 分担整张图片的编码，再汇总视觉特征交给原语言 TP2；同时用两个完整 TP1 副本和 vLLM 的 TP2/PP2 做方向比较。它没有实现 Prism PP，也没有把引擎改成异步 CPU/GPU 调度。

## 四种并行不要混着解释

| 方式 | 每张卡负责什么 | 可能改善什么 | 主要代价 |
|---|---|---|---|
| 语言 TP2 | 每层权重、Attention heads 与部分计算切分到两卡 | 单请求 Decode 时间、每卡权重/KV 占用 | 层间 collective 与控制开销 |
| Encoder DP + 语言 TP2 | ViT 权重仍各卡一份，图片按整图分工；语言模型保持 TP2 | 多图冷 Prefill 的重复视觉计算 | Main/DeepStack 聚合、分工不均、临时缓冲 |
| 两个 TP1 完整副本 | 每卡独立模型、Scheduler 和 Prefix Cache，各接不同请求 | 多请求总吞吐、负载下排队时间 | 两套完整权重与独立缓存，不加速一条孤立请求 |
| PP2 | 模型层分为前后两段，传中间 hidden states/residual | 有并发时的流水推进、较少的逐层 TP 通信 | 单请求串行经过两段，流水空泡和阶段不均衡 |

因此 Encoder DP 不是请求副本 DP，也不是把语言 TP2 关闭。PP 通信次数少，不代表一定更快：视觉集中在前段时，按语言层数平均分配也未必按耗时平均。当前 PP2 只是 vLLM 0.25.1 的 eager 方向参照。

## 图片从哪里分，到哪里重新合并

入口配置为 `vision_encoder_parallel_mode="replicated" | "data"`，默认仍为 replicated；可运行配置见 [tp2_vision_data.json](../configs/tp2_vision_data.json)。参数经 [config.py](../prism_infer/config.py) 和 ModelRunner 传到 [Qwen3VLModel._encode_images()](../prism_infer/models/qwen3_vl.py)。实际调用顺序是：

1. rank 0 的 Scheduler 形成同一个 BatchPlan、PrefillSlice 和 KV copy 计划。
2. ModelExecutor 先通过 TP 控制通道让各 rank 完成对应页复制，再广播 `run_plan`。
3. 各 rank 的 InputPreparer 按相同 slice 截取本轮图片，拼接原始 patch payload 和 grid。
4. `prepare_language_inputs()` 的 image 分支调用 `_encode_images()`；data 模式进入 [encode_images_data_parallel()](../prism_infer/vision/data_parallel.py)。
5. 汇总后的视觉特征按原 Prompt 顺序注入 image pad，再进入未改变的语言 TP2。

分工单位是一张完整图片，不是任意 patch 行。每张图 patch 数为 `T×H×W`；按 patch 数从大到小，把当前图片放到累计负载最小的 rank，同负载按 rank 顺序决定。它是简单的工作量近似，不宣称最优调度。分配后，每个 rank 内仍恢复图片原次序，并调用已有本地 Encoder/microbatch 路径。

以当前 Qwen3-VL-8B 为例，每张图经 spatial merge 后输出 `N_i=T_i×H_i×W_i/merge_size²` 行，每行为 4096 维。不能只收集主视觉输出：语言模型前几层还会使用全部 DeepStack 输出，漏掉它们即使形状能拼起来也改变模型计算。

设 `C=1+DeepStack层数`、rank r 的总输出行为 `N_r`、`N_max=max(N_r)`，实现把主输出与全部 DeepStack 一起打包：

| 张量 | 形状 | 用途 |
|---|---|---|
| 本 rank 的 packed | [C, N_max, 4096] | 有效行写入，剩余行 padding |
| all-gather 结果 | [TP, C, N_max, 4096] | 一次 collective 收齐各 rank 的结果 |
| 恢复后的 outputs | [C, sum(N_i), 4096] | 按原 image index 放回，而非直接按 rank 拼接 |

最后返回 `outputs[0]` 和其余 DeepStack tensors。图数少于 rank 数时，空 rank 不调用 Encoder，但仍参加同一 collective；不同 rank 的本地 microbatch 数可以不同，collective 不能塞进各自的 microbatch 循环。输出顺序由共同 grid 与 image index 决定，不依据 rank 0 私有的 Prefix hit 标志。

只有一张图片时，data 模式也走原 replicated 路径。这是根据实际聚合开销做的取舍：没有第二张图可分工，额外 all-gather 反而增加等待。视频仍直接调用原 `_encode_visual_payload()`，没有把单帧视频误判为 image，也未声称完成 Video Parallel。

当前仍把冷请求的原始媒体广播给所有 TP ranks，因此 Encoder DP 节省的是重复视觉计算，不是所有输入传输。每个 rank 仍持有完整 ViT 权重，聚合也需要额外缓冲；不能把此改动称为“视觉权重分片”或无条件省显存。

## Prefix 命中为什么必须一起改

Prefix 仍由 rank 0 的 BlockManager 决定；各 rank 在自己的语言 KV 分片中使用相同页号和逻辑范围。命中不需要重新聚合公共视觉输出：对应 PrefillSlice 已没有那些 image pad，输入准备自然不再把它们送进 Encoder。

这轮真实请求暴露并修复了三个连接处，代码分别在 [Sequence](../prism_infer/engine/sequence.py)、[TP 控制协议](../prism_infer/engine/tp_control.py) 和 [Executor](../prism_infer/engine/executor.py)：

| 现象或开销 | 原因 | 本轮改动 |
|---|---|---|
| TP worker 无法处理只剩后几张图的 Prefill | pickle 恢复时漏掉 `image_merge_size`，图片 span 无法重建 | 序列化并恢复该字段，检查恢复后的整图子集切片 |
| 热 Prefix 尾页 CoW 报 unsupported TP control method | Executor 已调用 `copy_kv_block_prefixes`，TPMethod 没登记 | 补上方法名；valid_rows 随命令传到每个 rank，并接收 ACK |
| 完整视觉 Prefix 已命中，Pipe 仍搬原始 pixel/grid | 旧序列化只看“是否 Prefill”，没看哪些视觉 token 仍待计算 | 当前剩余 Prefill 不再含该模态时，省略该模态 payload/grid |

最后一项不能改成“发送时只留下剩余几张图”。partial hit 下，worker 仍用完整 Prompt 的绝对 token span 定位图片；若提前缩短 grid，却不同时重写整个位置契约，会把后面的图片当成前面的图片。当前做法是：该模态还有任何待算 token，就保留完整原 payload/grid，交给 InputPreparer 按本轮绝对区间切片；只有整段模态已覆盖才省略它。

图片边界回退与 CoW 原则不变：block 命中停在图内时，cached 长度退到图起点；所有将重写的共享页先私有化，只复制仍需保留的前缀行。完整页仍在执行完成后发布，不能因为已经分配或已经广播就提前变为可命中。

独立的 Vision 输出缓存仍只支持 Qwen3-VL、TP1，这个已有约束现在会在模型构造前报错。TP 通道拓扑只在初始化时检查，timeout 在初始化或显式修改时检查；消息构造后不再在发送前重复验证字段。接收消息的验证、通道断开、ACK 的 request ID/rank 错配和超时处理仍保留。这次整理没有改动 Attention/KV 算子，也没有产生新的 GPU 性能数据。

## 数值差异查到了哪里

不等尺寸图片的整批/分片比较出现过非逐元素相同。为了区分通信与编码本身，把各分片放回同一 GPU、同一个 Encoder 顺序执行，去掉跨卡通信后仍能复现差异。因此不能直接归因于 all-gather 或恢复图片顺序错误。

在该受控复现中，patch embedding、position embedding、RoPE 和首层 QKV 的观测结果相同，首个差异出现在第一个 ViT block 的 Attention→输出投影观测点；提升为 FP32 后差异缩小。它支持继续调查计算形状与舍入路径，但当前观测点包含多个操作，尚不能断言某个 SDPA、GEMM 或具体 CUDA kernel 是根因。原始逐阶段记录见 [vision-partition-9763.json](../artifacts/vision_parallel_20260908/raw/vision-partition-9763.json)。

真实请求检查取现有 MuirBench materialization 中 20 条多图请求，按 Sample ID 对齐，比较首个 EOS 前的答案 token；没有 EOS 时比较到 16-token 上限。两种 Encoder 模式在这个范围内 token 一致，其中 19 条能读成有效选项，1 条未解析。原脚本为了固定输出长度继续生成 EOS 后文本，因此不能把旧 prediction 的解析统计或整段强制输出直接当作正常答案结论。

这不是全 MuirBench 准确率测量，也不是完整 hidden states/logits/概率分布逐元素相同的证明。配对规则和原始请求分别保存在 [summarize.py](../artifacts/vision_parallel_20260908/summarize.py)、[replicated 请求](../artifacts/vision_parallel_20260908/raw/muir-replicated-9765.json) 与 [data 请求](../artifacts/vision_parallel_20260908/raw/muir-data-9765.json)。

## TP、PP 与副本比较的实际边界

本次 5090 环境没有直接 GPU P2P 路径，但 TP2 collective 仍能执行；“无 P2P”不等于“TP2 不可跑”。它改变通信代价，最终应看完整请求而不是只按互联名称选择 TP 或 PP。该环境结论不能外推到另一台 5090 或 B300 的拓扑。

vLLM 的 `--encoder-mode data` 对应 `mm_encoder_tp_mode="data"`，是 TP ranks 间的 Encoder 数据分工；`weights` 才是其视觉权重 TP。它们都不能直接视作 Prism 原 replicated Vision 的同义配置。PP2 使用 TP1×PP2，Vision 只在首段执行；现有 36 层语言模型默认分为 18+18 层，DeepStack 的前三层注入仍由首段负责。

PP2 首次启动失败并不是模型不支持 PP。嵌套 `text_config` 缺少 architectures，构造语言子模型时重新进行 PP 类型检查，报 `No model architectures are specified`。当前 [vLLM 比较脚本](../benchmarks/bench_vllm_parallel.py) 仅通过以下参数补齐启动配置，不改模型权重或 vLLM 源码：

~~~python
hf_overrides={"text_config": {"architectures": ["Qwen3ForCausalLM"]}}
~~~

Prism 使用自己的 in-process token client，vLLM 使用 AsyncLLM；vLLM 方向比较默认 eager 且关闭 async scheduling，Prism 主测量可以使用 Decode CUDA Graph。两者的输入准备和缓存清理范围也有差别，所以这里不能从一张表推出 Prism 对竞品的通用绝对排名。

两个 TP1 副本的吞吐按“两份结果的总 token 数 ÷ 最早提交至最晚完成的共同时间跨度”计算，不把两份局部 tok/s 直接相加。cold/hot 分别等待两个副本就绪后启动；完整权重和 Prefix Cache 是各自一份。副本吞吐改善不代表单个请求 Decode 更快。

## 使用入口与作业范围

远端源码目录是 `/home/lcpu/87120912/Prism-Infer-vision-dp`，模型是 `/home/lcpu/87120912/models/Qwen3-VL-8B-Instruct`；Prism/vLLM 分别复用 `/home/lcpu/87120912/venvs/prism` 与 `/home/lcpu/87120912/venvs/vllm`。新增日志、缓存和临时数据都留在用户目录，主输出目录为 `/home/lcpu/87120912/vision-dp/runs`。

已有 [作业脚本](../artifacts/vision_parallel_20260908/) 指定 `--nodes=1 --gpus=2 --time=00:15:00`，并把 TMPDIR、Triton/Inductor/CUDA cache 指到该用户目录。同步当前工作树与作业脚本后，在登录节点提交即可，不在登录节点直接跑模型：

~~~bash
cd /home/lcpu/87120912/Prism-Infer-vision-dp
sbatch artifacts/vision_parallel_20260908/bench_2gpu.sh replicated
sbatch artifacts/vision_parallel_20260908/bench_2gpu.sh data
sbatch artifacts/vision_parallel_20260908/replicas_2gpu.sh
sbatch artifacts/vision_parallel_20260908/pp2_2gpu.sh
~~~

直接入口的参数如下；直接运行同样只能在已获得的 GPU 作业内进行：

| 脚本 | 关键参数和工作 |
|---|---|
| [bench_vision_parallel.py](../benchmarks/bench_vision_parallel.py) | `--tensor-parallel-size 2 --vision-mode replicated/data --execution-backend cuda_graph`；顺序冷/热、换问题、partial Prefix、并发请求 |
| [bench_vision_replicas.py](../benchmarks/bench_vision_replicas.py) | `--output-dir ...`；使用作业分配的两张卡，启动两个独立 TP1，分别同步 cold/hot 开始 |
| [bench_vllm_parallel.py](../benchmarks/bench_vllm_parallel.py) | TP2 用 `--tp 2 --pp 1 --encoder-mode data`，PP2 用 `--tp 1 --pp 2 --encoder-mode weights`；默认 eager |
| [check_vision_parallel_requests.py](../benchmarks/check_vision_parallel_requests.py) | `--materialized-root /home/lcpu/87120912/Prism-Infer/data/quality/materialized --vision-mode replicated/data --limit 20`；现有真实多图请求检查 |

请求脚本均需显式给 `--model` 和输出路径。八色图 fixture 与 `--images` 真实图片入口共用加载函数；默认顺序重复数和并发数是小规模设置，已有作业包装可覆盖它们，结果中的实际 engine_options/protocol 才说明本次负载。

两套 benchmark 和双副本入口都支持 `--image-max-pixels`，默认 `200704`（448²）。`--image-size` 只设置 fixture 原图边长；要增加实际视觉计算量，需同时提高像素上限。实际处理结果以每条请求的 `prompt_image_token_count` 为准，Prism 还记录 `image_grid_thw`，不再只按原图宽高判断负载。这个默认值只针对这些 benchmark，未改变 LLM 公共接口的默认图像配置。

每个已完成的请求批次都会保存报告，写盘在该批次的计时结束后进行。报告通过同目录临时文件替换，硬超时发生时至少保留最近一次完整保存；正在执行的批次不保证保留，也没有实现断点续跑。双副本监督器默认运行上限为 840 秒，为 15 分钟的 Slurm 作业留出 60 秒清理时间。

TTFT 从客户端提交前计到首 token；ITL 是客户端观察的 token 间隔，E2E 计到完成返回。vLLM 一次 yield 可能包含多个 token，原始 chunks 保留这一事实，不反推出不存在的逐 GPU step 时间。并发结果是有限请求批次、包含 Prefill 和排空过程，不是持续到达下的饱和吞吐。

顺序阶段只报告请求延迟，不再输出 tok/s 或 req/s；它的 `duration_s` 是包含阶段间隔的观察窗口。此前 vLLM 冷热交替运行导致顺序阶段的旧吞吐分母混入另一阶段时间，派生汇总现已从原始请求重算，旧 raw 不改写。下表使用的逐请求延迟和单个并发批次吞吐不受影响。

## 实测结果

Qwen3-VL-8B、两张 RTX 5090，八张 448×448 fixture 图片，greedy 16 tokens。Prism 使用 Scaled-FP8 KV、Decode CUDA Graph，TP2 的两卡合计 KV 为 1,245,708,288 bytes。顺序请求报告 5 次的中位数；并发结果为总计 8 请求的一次批次。模型加载、图片文件读取和预热不计入，Processor/请求准备计入。汇总由 [summary.json](../artifacts/vision_parallel_20260908/summary.json) 生成，原始文件与作业对应关系见[运行记录](../artifacts/vision_parallel_20260908/README.md)。

| 对照 | 指标 | 实测结果 |
|---|---|---|
| 原 replicated Vision → Encoder DP，只测视觉阶段，含聚合 | 两 rank 较慢值的中位数 | 59.33 → 31.81 ms，1.86× |
| 同一 TP2，replicated → data，作业 9770 | 冷请求 TTFT / E2E | 644.64 → 603.38 ms / 771.15 → 730.95 ms |
| 同一 TP2，replicated → data，作业 9770 | 8 请求冷批次吞吐 | 24.39 → 26.86 output tok/s |
| data 模式，省略已覆盖图片序列化前后，9764 → 9770 | 同问题完整 Prefix 命中 TTFT | 504.22 → 312.12 ms；不是 Encoder DP 本身带来的热命中收益 |
| TP2 data → 两个 TP1 副本，9770 / 9766 | 冷 / 热批次吞吐 | 26.86 → 68.75 / 66.04 → 93.85 output tok/s |
| vLLM 0.25.1 内部，TP2 encoder-data → PP2，eager，9761 / 9767 | 冷 TTFT；冷 / 热批次吞吐 | 209.17 → 294.09 ms；159.63 → 101.49 / 271.97 → 115.23 output tok/s |

不能把视觉阶段的 1.86×写成端到端 1.86×。省略已覆盖图片的序列化后，正向运行 9769 的冷 TTFT 为 655.60 → 517.43 ms，交换运行次序的 9770 为 644.64 → 603.38 ms；最初 9764 几乎没有冷 TTFT 差异。原始结果均保留，说明端到端还受 CPU 预处理、控制和运行波动影响，不把某一轮百分比当作普遍承诺。两个最终次序中的完整 Prefix 热命中约 0.30–0.32 s；9770 中两种 Vision 模式的热 TTFT 几乎相同，符合热请求没有 Vision 工作的预期。

两个 TP1 副本合计 KV 为 2,491,416,576 bytes，是这组 TP2 的两倍；两份完整权重也有额外显存成本。这是同样两张 GPU 的部署选择，不是同 KV 字节预算比较。对当前 8B、短输出并发工作，副本吞吐更合适；需要较低 Decode 延迟或更大的单引擎容量时，TP2 仍有意义。Prism 的两个副本目前由独立进程运行和基准客户端分配请求，没有实现统一路由、跨副本 Prefix Cache 或负载均衡服务。

PP2 参照没有显示出优势，因此本轮不新增 Prism PP。这个结论只适用于这里的 eager、短输出和批次设置，不代表长输出、更大模型或充分填满流水线时 PP 仍然更慢。也不能把上表 vLLM 行与 Prism 行直接比较：两套客户端、KV 精度、冷缓存行为及执行模式不同。

完整冷/热/partial 请求均正常完成；partial 的 A+B→A+C 请求复用了 512 个前缀 token，并与 APC-off 的 A+C 输出一致。20 条真实多图检查中，EOS 前或 16-token 上限内的序列 20/20 一致，有效选项 19/19 一致，两模式均 11/20 答对；强制 EOS 后生成的完整 16-token 轨迹只有 16/20 一致。单独的 [Trace](../artifacts/vision_parallel_20260908/trace.json) 用于看 CPU/GPU 与通信关系，不用带 profiler 的时延替换上述普通运行结果。

## 面试时怎样解释这项工作

“为什么不直接多开副本？”——副本解决请求之间的并行；这里先针对一条多图请求中重复执行的 ViT 工作，把图片分给已有 TP ranks。两种方式必须以相同 GPU 数、不同并发程度比较，不能互相替代。

“为什么只分整张图？”——图内是双向视觉 Attention，任意切 patch 会改变模型计算或引入新的通信；整图是现有独立 Attention segment，聚合后恢复原 image index，语言模型看到的输入顺序不变。

“为什么不说完全 exact？”——已经沿位置、RoPE、QKV 和 Attention 投影观测点做了排查，也做了真实请求配对；但特征存在形状相关数值差异，有限答案 token 一致不能证明所有概率都相同。

“你真正修改了什么？”——图片工作分配与特征聚合、单图取舍，以及多卡 Prefix 链上的字段恢复、控制命令登记、已覆盖媒体不再重复序列化。语言 TP2 与同步 `schedule → execute → postprocess` 没有被包装成新的异步或流水引擎。
