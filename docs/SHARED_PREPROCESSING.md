# 共享 CPU 媒体缓存与后台准备

这轮解决的是多模态请求到达模型之前的重复工作与线程阻塞。旧的 Processor 结果缓存主要属于 OnlineServingSession，普通生成和 HTTP 入口不能直接复用它；同图请求仍会重复经过 Processor，并重新计算处理后媒体的身份。与此同时，HTTP serving 的 owner 线程自己做 CPU 媒体准备，冷请求到来时会让正在输出的 Decode 一起等待。

现在，一个 engine 的几个入口共用 [MediaPreprocessingCache](../prism_infer/engine/media_preprocessing.py)。Serving 另用一个 CPU worker 准备媒体，完成后才由原 owner 提交请求。这里只并行了 CPU 准备：GPU 仍按同步 `schedule → execute → postprocess` 执行，没有实现 GPU 异步调度，也没有 Prefill/Decode 分离。

## 三种缓存，各自省掉什么

| 缓存 | 保存什么 | 省掉什么 | 没有省掉什么 |
|---|---|---|---|
| 本轮 engine 共享的 CPU 媒体缓存 | Processor 的 pixel/grid、相关 ImageInputs/VideoInputs、已计算的整组及逐图 identity | 相同媒体的 Processor 工作，以及重复计算处理后媒体 identity | 原始媒体内容读取、必要的问题 tokenization、每请求 M-RoPE/Sequence 构造 |
| 原有视觉输出缓存 | Vision Encoder 主输出与全部 DeepStack 特征 | 对相同视觉输入重新执行 ViT | 语言模型 Prefill 与问题后缀计算；本轮 HTTP 实验关闭此项 |
| 原有 Prefix KV Cache | 已完成的公共前缀 KV 页及页引用、逻辑/物理长度 | 公共前缀的语言前向，完整视觉前缀命中时还可跳过 ViT | 问题后缀、生成 token 的计算，以及共享尾页需要的 CoW |

“共享”指同一个 engine 的普通生成、OnlineServingSession 与 HTTP 入口共享，不是不同模型实例、进程或机器间共享，也不跨进程重启保留。CPU 缓存命中不自动等于 Prefix KV 命中：前者确认媒体处理结果可复用，后者还要确认连续左前缀、页就绪和驻留状态。

## entry 不只是 pixel/grid，也不是整条 Sequence

当前是一个按媒体内容、请求布局类型及 image marker 查找的 LRU，最多保留 128 个 entry；模型与 Processor 的语义 namespace 随 engine 建立。entry 保存处理后的媒体张量、已计算的 identity，以及最近一次兼容 Prompt 对应的 inputs。这个上限按 entry 数量计，不是固定 CPU 字节预算。

同媒体、同问题再次到达时，可以直接复用该 inputs，里面允许包含 `input_ids` 和 `attention_mask`，不需要为了“每次新请求”再 tokenize 一遍。换问题时，`_rebind_cached_media_prompt()` 对新旧模板做 tokenization，确认变化位于完整视觉占位区之后，再替换问题 token、构造新的 inputs，并保留 pixel/grid 和 identity。模板不匹配、改动触及视觉区域等不能安全重绑的情况，会回到完整 Processor 路径。

所以不能把实现描述成“缓存绝不含 input_ids”，也不能说“命中后每次仍要 tokenize”。真正每次新建的是请求状态：[LLMEngine._prepare_image_sequence()](../prism_infer/engine/llm_engine.py) 重新计算本请求 M-RoPE positions/rope_delta，再构造独立 Sequence。采样参数、输出 token、状态机和 block_table 不共享。

媒体对象本身可能可变。当前每次请求仍按其内容计算查找身份，不使用 Python 对象地址来假定“同一个 PIL 对象永远没变”。命中避免的是 Processor 及对其结果的重复身份计算，不是“再也不哈希”。entry 发布与 LRU 更新受短锁保护，内容读取和 Processor 不在锁内；并发冷 miss 允许各自计算，不引入另一套等待协议。

## 公共准备路径与 owner 交接

普通入口与 OnlineServingSession 最终调用同一个 [`_prepare_media_request()`](../prism_infer/engine/llm_engine.py)。图片的 `image/images` 名称在缓存入口统一处理；interleaved 图片和视频仍保留各自的布局语义。已经计算好的 `media_identity` 直接交给 Sequence 构造，不再在那里重复扫描同一份处理后像素。

HTTP 路径可以按以下链路逐段阅读：

| 执行位置 | 调用 | 产生的状态 |
|---|---|---|
| ASGI 入口 | `_to_generation_request() → runtime.submit()` | 网络数据变为请求；记录入口时间并占用有界容量 |
| serving owner | `_admit() → engine._allocate_request_id()` | 分配 engine ID，加入待准备队列；尚未查询 Prefix 或分配 KV |
| 单 CPU worker | `_prepare_media_request() → cache.prepare()` | 查媒体缓存；必要时 Processor/identity；构造本请求 M-RoPE 与 Sequence |
| serving owner | `_admit_prepared() → engine._submit_sequence()` | 查询 Prefix、交给 Scheduler；之后才有实际 GPU 工作和 KV 分配 |
| serving owner | `step_result() → TOKEN/DONE` | 继续原有模型执行与公开响应事件 |

实现见 [ServingRuntime](../prism_infer/serving/runtime.py)。网络协程不直接运行模型；CPU worker 也不查询 Prefix、修改 Scheduler、分配 KV 或决定采样。文本仍走 owner 的 `add_request()`，没有把所有入口改造成新的调度系统。

普通同步生成仍同步准备；OnlineServingSession 继续使用它已有的后台准备机制，但缓存不再由 session 独占。HTTP 的新 worker 直接进入现有 `_process_image_inputs()` 等函数，保留 `preprocess.image_processor`、M-RoPE 等区间，便于对照 CPU 准备与模型执行的时间位置。

## 单 Future、有界容量、取消和退出

ThreadPoolExecutor 只有一个 worker，并且 runtime 同时最多向它提交一个 Future。其他媒体请求留在 runtime 自己的待准备队列，不靠 executor 内部的无界队列吸收请求。`ingress_capacity` 统一覆盖入口队列、待准备请求和正在准备的请求；请求真正交给 engine 后才归 engine 的 admission 管理。

owner 每轮只处理入口和取消队列的有限快照，然后给模型一次推进机会，避免持续到达的冷请求无限延长一次 drain。没有 runnable 请求时等待 Event；提交、取消、CPU 完成和停止信号会唤醒 owner，不靠持续轮询耗满 CPU。

取消待准备请求时，直接从待准备队列移除并发一次 DONE；取消正在准备的请求时，也立即给客户端一次 DONE，但 CPU 操作不能被强行中断，因此容量要等 Future 完成才释放。其结果会被丢弃，不进入 `_submit_sequence()`，也不会分配 KV 或再次发 DONE。

CPU 准备异常只给该请求发 ERROR，不终止其他请求或 owner。shutdown 先结束活跃、排队和准备中的网络请求，再等当前 CPU 工作结束，最后调用 engine.exit，避免准备线程仍读 Processor/cache 时 owner 已经拆掉引擎。相关行为由 [FakeEngine/Event 测试](../tests/test_serving_runtime.py) 覆盖，包括 CPU 阻塞期间 Decode 继续、容量占用、排队取消和退出顺序。

`runtime.submit()` 保存的 `perf_counter_ns` 会传给媒体的 `_submit_sequence()`，文本也传给 `add_request()`。因此 engine 内部 TTFT 不会把后台准备时间悄悄排除。但 HTTP JSON/base64/图像解码发生在 runtime.submit 之前；本轮没有搬走这些工作，它们仍由下面的 HTTP 客户端计时包含。

## 固定 HTTP 请求上的结果

本次对照使用同一套 Qwen3-VL-8B-Instruct、TP1、RTX 5090、Scaled-FP8 KV 和 Decode CUDA Graph 配置：64 个 KV blocks、page size 256、最大模型长度 4096，Prefix Cache 开启，视觉输出缓存与 Chunked Prefill 关闭。配置见 [engine.json](../artifacts/shared_preprocessing_20260908/engine.json)。这不是上一轮 Encoder DP 实验。

输入是八张 448×448 纯色图片，不是 MuirBench。先用一个短请求 warmup/建立媒体组，再同图换三个问题；另外测一次独立长 Decode，以及三次“长 Decode 输出至少三个 token 后插入一组新颜色图片”。长请求固定生成 128 tokens，短请求固定 16 tokens，使用 greedy 且 `ignore_eos=True`。

首条 warmup 的时延受首次执行影响，不拿它计算优化收益。下面每格取对应三个请求的指标中位数：

| 观察对象 | 原 HTTP 路径：job 9788 | 共享缓存＋后台准备：job 9790 | 变化 |
|---|---:|---:|---:|
| 同图换问题的 TTFT | 317.964837 ms | 189.139526 ms | −40.52% |
| 被冷图插入的长 Decode：每请求 max ITL，再取中位数 | 388.486850 ms | 216.539206 ms | −44.26% |
| 插入的冷图请求自身 TTFT | 419.900482 ms | 434.612127 ms | **+3.50%，回退** |

第二行不是 p99，也不是把所有 token interval 混在一起求一个最大值。它先对每条长请求取最大的客户端 token 间隔，再对三条请求取中位数。TTFT 从 HTTP 请求发送前计到首个 SSE token，包含上传、服务端媒体解码、准备和排队；一次网络读取完成的多个 SSE 事件共享观察时间，不插值猜测 GPU token 时间。

数值来自 [HTTP baseline](../artifacts/shared_preprocessing_20260908/raw/http-baseline-9788.json) 与 [HTTP candidate](../artifacts/shared_preprocessing_20260908/raw/http-candidate-9790.json)，已用逐请求原始时间戳重算。按请求顺序同时核对 phase、Prompt、媒体组和生成长度后，11 条请求的 624 个完整输出 token 全部一致，两边的 SSE token 也均与各自最终 token 列表一致。这里包括 warmup 的输出一致性检查，但不把 warmup 纳入性能收益。

服务端 [baseline 记录](../artifacts/shared_preprocessing_20260908/raw/server-baseline-9788.json) 与 [candidate 记录](../artifacts/shared_preprocessing_20260908/raw/server-candidate-9790.json) 保存模型参数和内部指标。candidate 在这组请求结束时记录到 7 次 CPU 缓存命中、4 次 miss，其中 4 次换问题重绑；这支持复用路径确实发生，不等于已经把每一毫秒收益分别归因给缓存和后台线程。

## 为什么不能说“冷图也变快了”

后台准备让已有请求的 GPU Decode 可以在冷请求做 CPU 工作时继续推进，但冷请求自身的 Processor 工作并没有消失；准备好之后，它仍要等待 owner 调度，并执行原来的 GPU Prefill。CPU 阶段可以并行，不代表 GPU Prefill 不再打断 Decode。因此本次长 Decode 的最大停顿缩短了，冷图自己的 TTFT 却略有回退，两者必须一起报告。

这一对照同时包含共享缓存和后台准备，不能仅凭总体时延差把二者收益相加或拆分。它也没有证明任意媒体、任意负载都改善，更不是全 MuirBench 质量结果或跨引擎排名。固定长度输出相同，只说明这 11 条请求在该协议下没有观察到 token 差异。

## 如何复现与定位

[run_http.sh](../artifacts/shared_preprocessing_20260908/run_http.sh) 通过 `baseline/candidate` 参数选择源码目录，并在单 GPU、最长 15 分钟的 Slurm 作业内启动正常 HTTP server 与同一个客户端。服务器 readiness 通过 health 检查确认，不把模型启动混进请求指标。

当前远端实验根目录为 `/home/lcpu/87120912/preprocessing`，两份源码位于其 `baseline/` 和 `candidate/`；模型与环境分别复用 `/home/lcpu/87120912/models/Qwen3-VL-8B-Instruct` 和 `/home/lcpu/87120912/venvs/prism`。新增输出、临时文件和 cache 路径均留在该用户目录。不要在登录节点直接运行模型，也不要覆盖已有 job 对应的 raw。

客户端入口是 [bench_serving_preprocessing.py](../benchmarks/bench_serving_preprocessing.py)，提供 `--base-url`、`--label`、`--repeat`、`--long-tokens` 和 `--output`；服务入口是 [serve_preprocessing_study.py](../benchmarks/serve_preprocessing_study.py)，提供模型、配置、端口和服务端报告输出。实验文件与时间线说明集中在 [实验记录](../artifacts/shared_preprocessing_20260908/README.md)。

单独的 Nsight 运行记录到：首个注入冷图的 CPU Processor 区间为 115.02 ms，其中 GPU 计算活动覆盖约 100.50 ms，owner 在该区间发起了 10 次 Decode replay。两个 CPU 线程不同，GPU kernel 也确实在该时间窗口内执行，不只是把工作放进了队列。可打开 [trace.json](../artifacts/shared_preprocessing_20260908/trace.json) 查看。这个带 profiler 的运行只证明执行关系，不替代前面的普通运行时延。

HTTP JSON 解析和初始图片解码仍由原 ASGI 适配层处理，本轮移出的主要是 HF Processor 与请求准备。大体积媒体的上传、解码开销不能由本次 448×448 fixture 推断。CPU 缓存还会持有额外的 host Tensor 内存，当前按 128 个 entry 管理，未建立固定 CPU 字节预算。

面试中可以沿一条具体链路说明这项改动：重复媒体首先复用 CPU 结果；新问题仍建立自己的 token/位置/Sequence；冷媒体准备不再占住 GPU owner；完成后才由 owner 查询 Prefix 和运行模型。把可复用的数据、每个线程修改的状态、取消时的容量释放，与对应代码和请求记录一起讲清楚。
