# Prism-Infer 性能报告

> 更新日期: 2026-07-15
> 当前阶段: P6.12-B content-aware visual pruning quality research
> 报告性质: 包含 dirty validation 与 clean formal evidence；每节单独标注结论边界

## 1. 结论边界

本报告的第一组数据用于验证 P6 benchmark schema、runner 和四种 internal mode 能在同一输入条件下工作。运行时仓库为 `git_dirty=true`，commit 为 `f4bf51a7c054df612dfec6148a2667537953c6a4`，因此数据不能作为 clean-commit 发布结果，也不能用于声称超过外部框架。

当前只覆盖一个 synthetic 单图 case、batch/concurrency 1 和 8-token greedy 输出。它能验证计时与结果采集链路，不能替代真实图片质量集、长上下文、并发容量、online serving 或 vLLM/SGLang/vLLM-Omni 对比。

## 2. Benchmark Contract

实现入口:

- `prism_infer/analysis/benchmark_schema.py`: versioned workload/result schema、统计与硬校验。
- `benchmarks/workloads/p6_internal_smoke.json`: text、单图、多图、视频和 mixed batch deterministic manifest。
- `benchmarks/bench_system.py`: Prism internal offline closed-loop runner。
- `tests/test_benchmark_schema.py`: schema 正向与缺失证据、hash、统计顺序负向测试。

每条 JSONL 记录包含环境和 commit、模型/config、execution/attention/compression mode、输入 shape 与 token 数、offline traffic 口径、warmup/repeat、同步计时边界、token IDs/hash、TTFT/ITL/E2E、request/token throughput、allocated/reserved/peak memory 和物理 KV cache bytes/capacity。

设计选择是把 `request_rate_per_s` 显式记为 `null`，同时标记 `offline_closed_loop`。当前 runner 没有 arrival process，不能把离线批处理结果伪装成 online request-rate benchmark。

## 3. P6.1 Runner Validation Baseline

### 3.1 环境与输入

```text
GPU: NVIDIA GeForce RTX 5090
CUDA: 12.8
Python: 3.12.3
PyTorch: 2.6.0a0+ecf3bae40a.nv25.01
Transformers: 5.13.0
Model: Qwen3-VL-8B-Instruct snapshot 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
Model dtype: bfloat16
TP: 1
Input: 1 x synthetic RGB 448x448, prompt tokens 210, image tokens 196
Output: max_tokens=8, temperature=0, ignore_eos=true
Traffic: offline closed-loop, batch=1, concurrency=1
KV config: 16 blocks x 256 tokens, capacity 4096 tokens
Warmup/repeat: 1/3
Timing boundary: torch.cuda.synchronize() before and after every engine step
```

运行命令:

```bash
PYTHONPATH=/data/Prism-Infer HF_HUB_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_system.py \
  --model /data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --case single_image_448 \
  --modes off_eager,off_graph,visual_prune,fp8_kv \
  --max-tokens 8 --warmup 1 --repeat 3 \
  --num-kvcache-blocks 16 --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --output data/p6_system/single_image_internal_baseline_20260711.jsonl
```

原始 JSONL 位于 `data/p6_system/single_image_internal_baseline_20260711.jsonl`。`data/` 默认不入库；正式实验需要保留命令、commit 和原始产物摘要。

### 3.2 Latency 与 Throughput

以下每个单元格均为 `median / p90 / min / max`。TTFT 包含 preprocessing；decode step 是 21 个同步 step 样本的 ITL 汇总；其余指标来自 3 次 measured repeats。

| Mode | E2E TTFT (ms) | Decode step (ms) | E2E latency (ms) | Engine output tok/s |
|---|---:|---:|---:|---:|
| `off_eager` | 55.433 / 70.201 / 54.731 / 70.201 | 30.492 / 30.573 / 30.258 / 34.183 | 272.862 / 283.261 / 267.831 / 283.261 | 30.080 / 30.586 / 29.070 / 30.586 |
| `off_graph` | 84.222 / 121.541 / 67.665 / 121.541 | 17.432 / 17.535 / 17.394 / 17.683 | 206.688 / 243.570 / 189.881 / 243.570 | 40.017 / 43.604 / 33.718 / 43.604 |
| `visual_prune` | 87.745 / 100.839 / 84.583 / 100.839 | 100.544 / 102.447 / 100.020 / 103.976 | 793.098 / 811.428 / 790.414 / 811.428 | 10.179 / 10.205 / 9.943 / 10.205 |
| `fp8_kv` | 453.590 / 455.524 / 445.913 / 455.524 | 35.213 / 35.635 / 35.079 / 35.793 | 700.031 / 703.113 / 693.199 / 703.113 | 11.541 / 11.654 / 11.515 / 11.654 |

### 3.3 Memory、KV 与 Correctness

| Mode | Allocated / reserved / peak median (MB) | KV bytes | KV byte ratio | Token exact vs `off_eager` |
|---|---:|---:|---:|---:|
| `off_eager` | 17319.02 / 19938.00 / 19698.23 | 603979776 | 1.0 | PASS |
| `off_graph` | 17328.06 / 19948.00 / 19707.27 | 603979776 | 1.0 | PASS |
| `visual_prune` | 17328.05 / 19944.00 / 19707.26 | 603979776 | 1.0 | PASS |
| `fp8_kv` | 17040.05 / 19656.00 / 19419.26 | 301989888 | 0.5 | PASS |

四种 mode 的 repeat 输出和相对 baseline 输出均为:

```text
[785, 2168, 3897, 374, 264, 6437, 11, 13794]
```

这里的 exact match 只说明该 synthetic 8-token greedy case 没有产生 token 差异，不代表完整质量评测。`visual_prune` 仍是 logical retained-KV view，因此 KV bytes 没有下降；不能将其称为 physical compaction。`fp8_kv` 在固定 16-block cache 下把 KV storage bytes 降为 0.5x，但本次 TTFT 和吞吐明显差于 BF16 baseline。

`off_graph` 的 decode-step latency 低于 `off_eager`，但 TTFT 更高。`visual_prune` 和 `fp8_kv` 的慢路径根因尚未经过 profiler 归因；P6.2 之前不把现象直接归因到某个 kernel、同步点或 framework overhead。

## 4. P6.1 初始记录的限制与下一门禁

- 本节 P6.1 初始记录为 dirty-worktree runner validation；提交后需要同命令 clean rerun。
- 只测了 synthetic single-image；manifest 其余 text/multi-image/video/mixed cases 尚未形成 P6 baseline 表。
- 只测 8-token 输出；长 decode、长 visual context、batch/concurrency matrix 和 OOM boundary 尚未测量。
- 该初始记录当时没有真实任务 quality metric、teacher-forced logits/ppl 或稳定前缀矩阵；后续 reference-task evidence 见 5.19。
- 没有外部 framework adapter 和公平对比结果。
- 没有 GPU utilization、kernel count、CPU launch gap 或分模块 trace；这些属于 P6.2。

P6.2 分层 profiling 已完成第一轮瓶颈定位，结果见下一节。当前仍需 clean-commit rerun，并补 RTX 5090 SM utilization；在此之前不形成正式性能 claim。

## 5. P6.2 分层 Profiling

### 5.1 方法与口径

新增 semantic collector 同时记录 CPU `perf_counter`、CUDA Event 和 `prism::*` NVTX range。CPU 时间用于观察 Python 与同步等待，CUDA Event 用于观察当前 stream 上的 inclusive GPU 时间；嵌套 region 不能相加。collector 默认关闭，profiling iteration 与 unprofiled benchmark repeat 分开运行。

Nsight Systems 使用 CUDA Profiler API capture range，只捕获额外的 profiled iteration。`.nsys-rep` 导出为 SQLite 后，由 `benchmarks/analyze_nsys.py` 读取 NVTX/CUPTI 表，并通过 runtime `correlationId` 关联异步 GPU activity。

当前 workload 与 P6.1 相同: synthetic 单图 `448x448`、prompt/image tokens `210/196`、输出 8 tokens、batch 1、1 prefill + 7 decode steps。所有 profiled outputs 与对应 unprofiled outputs token exact。所有结果仍为 `git_dirty=true`。

### 5.2 Semantic Region 结果

| Mode | Prefill `engine.model_runner` CUDA | Decode `engine.model_runner` CUDA/step |
|---|---:|---:|
| `off_eager` | 90.252 ms | 37.213 ms |
| `off_graph` | 58.803 ms | 17.737 ms |
| `visual_prune` | 59.162 ms | 116.964 ms |
| `fp8_kv` | 431.488 ms | 51.229 ms |

这些绝对值包含 CUDA Event、NVTX 和 `record_function` overhead，不替代第 3 节 unprofiled benchmark。它们的用途是比较同一 profiled iteration 内的 inclusive/subregion 组成。

`visual_prune` 每个 decode step 的关键组成:

| Region | CPU/step | CUDA/step |
|---|---:|---:|
| retained-index | 3.006 ms | N/A |
| retained paged gather | 68.463 ms | 69.055 ms |
| retained SDPA | 2.155 ms | 2.554 ms |

context length 为 `211..217`，retained length 为 `113..119`。gather 明确是当前 logical pruning 的第一优化目标，不能把当前退化归因给 SDPA 本身。

FP8 decode 每个 step 的关键组成:

| Region | CPU/step | CUDA/step |
|---|---:|---:|
| eager KV store | 4.534 ms | 5.042 ms |
| full-context gather | 3.538 ms | 4.008 ms |
| FP8 -> BF16 dequant | 2.068 ms | 2.420 ms |
| SDPA | 2.228 ms | 2.620 ms |

FP8 prefill 更严重：36 层 eager KV store CUDA 合计 `373.769 ms`，占 `model.language_model` CUDA `412.194 ms` 的主要部分。

### 5.3 Nsight Kernel/API 证据

以下为每个 capture 的 decode-step median。`off_graph` 的 explicit kernels 不包含 graph 内部 node，graph 本体时间单列。

| Mode | Explicit kernels | Graph launch | Async memcpy | Stream sync | Kernel busy | Graph execution |
|---|---:|---:|---:|---:|---:|---:|
| `off_eager` | 2004 | 0 | 10 | 2 | 16.137 ms | 0 |
| `off_graph` | 11 | 1 | 14 | 2 | 4.074 ms | 12.896 ms |
| `visual_prune` | 2148 | 0 | 3610 | 3566 | 17.384 ms | 0 |
| `fp8_kv` | 2220 | 0 | 298 | 110 | 16.112 ms | 0 |

归因证据:

- 252 个 `attention.decode.visual_prune.gather` ranges 合计 24696 次 async memcpy 和 24696 次 stream synchronize，即每个 layer/decode gather 98 次。当前 retained gather 中逐 segment 读取 `block_ids[...].item()`，与该计数一致。
- 288 个 `attention.kv_store.eager` ranges 合计 7812 次 stream synchronize，等于 `36 * (210 prefill + 7 decode)`。当前 FP8 store 对每个 token 执行 `slot_mapping[i].item()` 后分别写 K/V。
- `off_eager` 每 decode step 有 2004 次显式 kernel launch；`off_graph` 使用一次 graph launch，graph execution median `12.896 ms`。这支持继续保留 CUDA Graph 作为 decode 执行基线，但不直接证明 `torch.compile` 一定收益。

### 5.4 长 Context Paged Decode

batch=1、Qwen GQA shape `32 heads / 8 KV heads / head_dim 128`、BF16、warmup/repeat `10/50`:

| Context | Max diff | Prism Triton median | PyTorch reference median | 结论 |
|---:|---:|---:|---:|---|
| 256 | 1.953125e-03 | 0.0387 ms | 0.1352 ms | PASS，Prism 更快 |
| 1024 | 9.765625e-04 | 0.0873 ms | 0.1473 ms | PASS，Prism 更快 |
| 4096 | 4.882812e-04 | 0.2674 ms | 0.2071 ms | PASS，当前 Prism 更慢 |

因此不能声称现有 paged kernel 对所有 context 都优于 reference。context=4096 必须进入后续 kernel tuning matrix。

### 5.5 决策与剩余风险

基于本轮实测，第一个优化闭环应是:

1. 用 tensorized slot/block mapping 替换 visual gather 中逐 retained segment `.item()`，消除每 layer 98 次同步。
2. 为 FP8 cache 实现向量化 store，移除逐 token Python loop、`.item()` 和独立 K/V assignment。
3. 在 correctness PASS 后分别重跑 semantic profile、Nsight 和 unprofiled benchmark，不能只报告 kernel microbenchmark。
4. physical compaction 继续作为主线，但不应把当前 pathological logical gather 带入新 layout。

当前 Nsight 对 RTX 5090 GPU metrics 返回 `Already under profiling`，因此没有真实 SM utilization。报告中的 `kernel busy` 是 kernel duration 汇总，不是 GPU utilization。Megakernel 仍不启动；eager kernel 数量说明 launch 很多，但 CUDA Graph 已经是必须保留的对照，是否需要 persistent/megakernel 还缺少更长 decode 与 SM utilization 证据。

### 5.6 Visual Retained Gather 优化闭环

实现选择是在 `ModelRunner.prepare_decode()` 中，根据 CPU `Sequence.block_table` 把 retained logical indices 一次映射为 GPU int64 physical slots，并让 36 层 attention 复用。attention 对 flat canonical KV cache 执行 K/V 两次 `index_select`。

拒绝的替代方案:

- 每层把 logical indices 传到 GPU 后重新计算 block ordinal/offset：会重复 36 次 mapping kernel。
- 在每层保留 segment loop 但批量 `.tolist()`：仍把 GPU block table 拉回 CPU，而且不能消除 Python segment dispatch。
- 直接称为 physical compaction：当前 KV cache 和 block table 没有移动或释放，KV bytes 仍为 `603979776`。

同轮 unprofiled single-image benchmark:

| Mode | Decode-step median | Engine output tok/s | KV bytes | Token exact |
|---|---:|---:|---:|---:|
| `off_eager` | 30.834 ms | 26.183 | 603979776 | PASS |
| `visual_prune` tensorized | 33.529 ms | 27.412 | 603979776 | PASS |

这里 engine output tok/s 受到该轮 off prefill 波动影响，decode-step 是本次优化的主要指标。visual/off decode ratio 为 `1.087x`，尚未达到研究目标 `<=1.05x`。

before/after evidence:

| Metric | Segment gather | Tensorized slot gather | Change |
|---|---:|---:|---:|
| Unprofiled visual decode median | 100.544 ms | 33.529 ms | -66.7% |
| Semantic gather CUDA/step | 69.055 ms | 3.108 ms | -95.5% |
| Gather target async memcpy | 24696 | 0 | eliminated |
| Gather target stream sync | 24696 | 0 | eliminated |
| Whole-step stream sync median | 3566 | 2 | baseline level |

semantic profile 本身有 instrumentation overhead；before/after unprofiled 数据也来自同一 dirty worktree 的不同运行，所以这里只形成 focused optimization evidence，不升级为正式发布 claim。

质量 smoke 覆盖 text、single-image、multi-image 和 video，visual-prune 对 off 为 aggregate `32/32` token exact。physical KV bytes 没有下降，说明优化没有偷偷改变 P6.4 的完成定义。

剩余差距来自 retained gather、GQA expand 和 SDPA 的多算子执行。下一步不应恢复 Python gather，而应在 P6.5 设计 retained-slot-aware paged attention。

### 5.7 FP8 Vectorized KV Store 优化闭环

实现选择是复用现有自实现 Triton KV store。CUDA FP8 cache 不再进入 `_store_kvcache_eager()` 的逐 token loop；同一个 Triton launch 读取 slot mapping，同时把 K/V 写入 canonical paged cache，destination pointer dtype 完成当前 unit-scale E4M3FN 转换。CPU、无 Triton 环境和 correctness reference 继续保留 eager 实现。

选择依据来自本项目源码与实测，没有引用外部框架实现。拒绝的替代方案:

- PyTorch `index_copy_`：K/V 至少需要两个独立调用，并且 FP8 conversion/dispatch 边界不如现有单个自实现 kernel 清晰。
- 新增 CUDA extension：现有 Triton kernel 已在 Qwen shape 上证明 FP8 cast exact，额外编译系统和维护成本没有对应收益。
- 同时修改 FP8 attention：会让 store 与 gather/dequant 的收益无法独立归因，因此留到 P6.5。

Qwen prefill focused correctness 使用 key/value `[210,8,128]`、cache `[3,256,8,128]`、非连续 physical slots 和 `-1` padding。Triton 与 eager reference 的完整 K/V cache max diff 都是 `0.000000e+00`，untouched slot 保持原值。

before/after evidence:

| Metric | Eager FP8 store | Triton FP8 store | Change |
|---|---:|---:|---:|
| Semantic prefill store CUDA, 36 layers | 373.769 ms | 0.606 ms | -99.84% |
| Profiled language-model prefill CUDA | 412.194 ms | 35.627 ms | -91.36% |
| Nsight target kernels | 15624 | 288 | -98.16% |
| Nsight target async memcpy | 23436 | 0 | eliminated |
| Nsight target stream sync | 7812 | 0 | eliminated |
| Nsight target kernel time total | 15.192 ms | 0.286 ms | -98.11% |

同轮 unprofiled single-image 结果:

| Mode | Decode-step median | KV bytes | Token exact |
|---|---:|---:|---:|
| `off_eager` | 31.077 ms | 603979776 | PASS |
| `fp8_kv` vectorized store | 35.865 ms | 301989888 | PASS |

FP8/off decode ratio 仍为 `1.154x`。这是合理的边界结果：decode 每层只写一个新 token，移除 store 同步后，full-context gather、FP8->BF16 dequant、GQA expand 和 SDPA 仍占主要路径。整步 decode stream sync 从 `110` 降到 `74`，说明仍需消除 context-length host read，并实现 FP8-aware paged attention，而不是继续微调 store。

质量矩阵覆盖 text、single-image、multi-image 和 video，FP8 对 off 为 aggregate `32/32` token exact，KV bytes ratio `0.5`。所有数字仍来自 `git_dirty=true` focused experiment；unprofiled prefill 有明显波动，正式发布前必须在 clean commit 和开放 hardware counter 的平台复跑。

### 5.8 P6.3-A CUDA Graph Execution Matrix

统一 runner 升级为 benchmark schema v2，新增 batch/output matrix 和 execution backend contract。v2 明确记录 Graph capture scope/time、captured buckets、selected batch/padding，以及 synthetic request replication；validator 继续接受历史 v1 raw records。`ModelRunner` 内部单独测量 capture time，并修复 replay NVTX 曾把 max placeholder batch 误记为 selected graph batch 的问题。

设计选择与边界:

- 扩展统一 `bench_system.py`，不继续维护独立 CUDA Graph benchmark 的第二套统计口径。
- 单请求 manifest case通过完整 request replication 形成 batch，并记录 source count/factor；拒绝在 JSON manifest 中复制大量近似 case。
- capture time 在 `capture_cudagraph()` 内同步计时；拒绝用包含模型加载、warmup 和 KV allocation 的整个 LLM initialization time 代替。
- 本轮固定 `compression=off`，不把 visual/FP8 路径与 execution backend 混合归因。

single-image `448x448`、prompt/image tokens 每请求 `210/196`、16 个 256-token blocks、warmup/repeat `2/5`。12-cell unprofiled结果:

| Batch | Output | Eager decode | Graph decode | Speedup | Graph decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 30.704 ms | 17.460 ms | 1.759x | 57.255 |
| 1 | 32 | 30.746 ms | 17.469 ms | 1.760x | 57.246 |
| 1 | 128 | 30.739 ms | 17.623 ms | 1.744x | 56.742 |
| 2 | 8 | 31.681 ms | 17.688 ms | 1.791x | 112.966 |
| 2 | 32 | 31.636 ms | 17.698 ms | 1.788x | 113.001 |
| 2 | 128 | 31.682 ms | 17.834 ms | 1.777x | 112.130 |
| 4 | 8 | 31.651 ms | 18.207 ms | 1.738x | 219.327 |
| 4 | 32 | 31.662 ms | 18.213 ms | 1.738x | 219.552 |
| 4 | 128 | 31.668 ms | 18.349 ms | 1.726x | 217.998 |
| 8 | 8 | 31.779 ms | 18.820 ms | 1.689x | 423.984 |
| 8 | 32 | 31.829 ms | 18.828 ms | 1.691x | 424.628 |
| 8 | 128 | 31.856 ms | 18.956 ms | 1.681x | 422.096 |

24 records全部 repeat-stable；12/12 Graph cells 对 eager token exact。batch8/output128 生成 `1024` tokens，并把 16-block KV capacity 用到当前 workload 边界，没有 OOM 或 block mapping错误。

Graph capture time median 随 max batch/buckets 为:

| Max batch | Captured buckets | Capture median |
|---:|---|---:|
| 1 | `[1]` | 251.950 ms |
| 2 | `[1,2]` | 563.718 ms |
| 4 | `[1,2,4]` | 926.789 ms |
| 8 | `[1,2,4,8]` | 1317.755 ms |

batch8/output32 Nsight 验证了机制而不仅是 latency：eager 每 decode step 有 `2077` 个显式 kernels；Graph 路径有 `13` 个 graph 外 kernels、`1` 次 graph launch 和 `14.818 ms` graph execution。两者 stream sync 都是 `2`。Prefill 两边均为 `3617` kernels，因此当前 capture scope严格是 `decode_model_forward`，compute logits、sampler 和 engine preparation仍在 Graph 外。

Graph speedup在 output `8/32/128` 上稳定，但随 batch 从2增至8略下降。没有 hardware counter，不能确定这是 Graph 内 kernel效率、memory pressure 还是 graph 外 fixed cost变化；下一步 `torch.compile` preflight 必须先报告 graph break/guard/recompile，再与该 Graph baseline同条件比较。

限制是当前 performance matrix 使用 replicated synthetic single-image offline batch。mixed text/image/video batch=3 -> graph bucket4 已做 token exact correctness，但尚未做 padding性能矩阵；结果不能外推为 online serving 或所有多模态 workload吞吐。

### 5.9 P6.3-B Torch Compile Preflight

诊断先修复了一个测量基础设施问题：默认关闭的 semantic `profile_region()` 仍访问 `ContextVar`，把 decoder layer 切成 `6 graphs/5 breaks`。编译捕获期间返回标准 `nullcontext` 后，同一真实 layer 变为 `1 graph/0 break`。这项修改不改变 eager/profile collector 数值，focused test exact。

候选边界结论:

- 完整 language-model decode 虽为单 graph，但 2991-node cold compile 在 batch1/4 均把 RTX 5090 32GB 用尽，无法得到 steady latency。
- 完整 VisionEncoder 的 grid geometry 产生 6 个 graph breaks；拆出稳定 blocks/mergers tensor region 后 graph break清零，但 default/emulate-casts 的 27 层输出 max diff 仍为 `0.859375/0.515625`，不能接入。
- decoder RMSNorm 在 eager-cast 模拟下 exact 且 micro更快；MLP exact 后反而更慢；attention 单步 exact 且有 micro收益，因此进入 system ablation。

output32、warmup/repeat `2/5` 的统一 system matrix显示，attention-only compile 对 eager 的 decode speedup为 `1.43x..1.46x`，但 CUDA Graph 仍快 `1.20x..1.27x`。compile first decode为 `1.70..2.20 s`。batch1/4 token exact，batch2/8 所有行在 token 28 分叉；额外开启 `force_same_precision` 仍未修复。故 `off_compile_attention` 只保留为显式 unsafe benchmark reproduction，supported execution backend继续是 eager与 CUDA Graph。

该结论比“compile 有约 1.4x 加速”更重要：micro/短输出 correctness不能替代长输出系统门禁，且 compile收益不能与 CUDA Graph之外的弱 baseline单独比较。

### 5.10 P6.4 Visual KV Physical Compaction

P6.4 在 `Sequence` 的逻辑 token 序列与 attention 实际读取的 physical KV 之间增加显式 layout descriptor。prefill 完成后先跨 K/V 和 36 层把 retained slots gather 到临时 tensor，再覆盖页表前缀；只有 GPU copy 成功后才提交新页表并释放尾页。M-RoPE 继续使用 logical length，decode attention/context 和新 KV 写入使用 physical length/tail。

选择页表前缀原地复用，是因为它可以直接释放尾页且保留现有 paged kernel 接口。拒绝保留原 physical slots 的 sparse table，因为它无法回收中间/尾部页；也拒绝逐层原地移动，因为 source/destination 重叠会覆盖尚未读取的数据。该设计依据来自本项目 block manager 与 canonical KV layout，没有使用外部框架实现。

multi-image `2x448`、keep=0.5、output8、warmup/repeat `2/5`：

| Mode | Logical/physical prompt | Active blocks | Occupied KV bytes | Decode median | Quality |
|---|---:|---:|---:|---:|---|
| off | 408 / 408 | 2 | 75,497,472 | 32.204 ms | baseline |
| visual compact | 408 / 212 | 1 | 37,748,736 | 32.231 ms | first 6 exact, diverges at token 7 |

layout、swap/pickle、hash cleanup、跨页 append 和 M-RoPE focused regression 为 `64 passed`。mixed text/image/video smoke 证明 text row 为 dense no-op，image/video physical prompt 分别为 `112/226`。该阶段证明真实 page/byte reduction，但没有 TPOT 加速；质量与 keep-ratio Pareto 留到 P6.6。

### 5.11 P6.5 FP8-Aware Paged Decode

现有自实现 Triton online-softmax kernel 已通过 probe 证明能从 E4M3FN pointer 逐元素 load 并转 FP32 累积，因此 P6.5 复用同一地址计算和 softmax，不创建重复 FP8 kernel。engine FP8 decode 从逐序列 gather、整段 dequant、GQA expand 和 SDPA 改为 paged kernel 内 load/dequant；CPU/no-Triton 路径保留为显式 correctness reference。

Qwen GQA BF16/FP8 32-case matrix全部通过，max diff `<=0.00390625`。warmup/repeat `10/50` 的代表结果：

| Cache | Batch/context | Paged kernel | Eager reference | Ratio |
|---|---:|---:|---:|---:|
| BF16 | 1 / 4096 | 0.2701 ms | 0.2077 ms | 1.300x |
| BF16 | 8 / 4096 | 0.4527 ms | 1.6259 ms | 0.278x |
| FP8 | 1 / 4096 | 0.2232 ms | 0.2338 ms | 0.955x |
| FP8 | 8 / 4096 | 0.2602 ms | 1.8029 ms | 0.144x |

full-engine single-image output32、warmup/repeat `2/5` 中，off/FP8 decode median 为 `32.065/31.960 ms`，32-token exact，KV bytes ratio `0.5`。系统收益只有约 `0.3%`，说明 kernel 已消除旧 FP8 慢路径，但 batch1 短 context 下 attention kernel 不是主导整步 TPOT。所有数据仍为 dirty-worktree 内部证据，P6.10 需要 clean commit 重跑。

### 5.12 P6.6 Quality-Memory-TPOT/Capacity Pareto

P6.6 增加 `visual_compact_fp8` 正交组合：prefill 后先按 retained slots 做 physical compaction，再以 E4M3FN 保存物理页，decode 直接复用 P6.5 FP8 paged kernel。PyTorch CUDA 不支持这里需要的 FP8 `index_select/index_copy`，所以 compaction 使用两阶段 Triton copy（source -> temporary -> destination）避免重叠覆盖；CPU/BF16 保留 PyTorch independent reference。focused FP8 retained K/V max diff 为 `0`。

设计选择与边界：

- logical pruning 与 physical compaction 分开报告；前者不释放页且 retained eager backend 会影响 mixed text row，不能拿来证明 KV 容量收益。
- occupied bytes 按 active block page 粒度统计，而不是按 retained token 数线性估算。因此 single-image `210 -> 112` 仍在同一 256-token page 内，BF16 active bytes ratio 仍为 `1.0x`。
- quality 使用逐请求 greedy stable prefix，不把 FP8 数值扰动偶然把某个 token flip 回 baseline 解读为质量改善。
- 固定真实图片通过 path、source URL、SHA256、width/height 契约加载；资产不存在或内容变化会显式失败。

synthetic output128 代表点：

| Workload / keep | Mode | Logical/physical | Blocks | Active bytes | Stable prefix | TPOT ratio |
|---|---|---:|---:|---:|---:|---:|
| multi-image / 0.50 | compact BF16 | 408 / 212 | 1 | `0.5x` | `[6]` | `0.999x` |
| multi-image / 0.50 | compact FP8 | 408 / 212 | 1 | `0.25x` | `[27]` | `0.996x` |
| multi-image / 0.75 | compact FP8 | 408 / 310 | 2 | `0.5x` | `[60]` | `1.006x` |
| video / 0.50 | compact FP8 | 422 / 226 | 1 | `0.25x` | `[14]` | `1.002x` |
| mixed / 0.50 | compact BF16 | 638 / 344 | 3 | `0.75x` | `[128,20,14]` | `1.003x` |
| mixed / 0.50 | compact FP8 | 638 / 344 | 3 | `0.375x` | `[7,28,14]` | `1.004x` |

固定 COCO `000000039769.jpg`（原图 `640x480`，processor 后 300 image tokens）、output32、warmup/repeat `1/3`：

| Keep | Mode | Physical prompt | Active bytes | Stable prefix | TPOT ratio |
|---:|---|---:|---:|---:|---:|
| 0.25 | compact BF16 | 91 | `0.5x` | `[3]` | `0.995x` |
| 0.25 | compact FP8 | 91 | `0.25x` | `[3]` | `1.000x` |
| 0.50 | compact BF16 | 166 | `0.5x` | `[3]` | `1.001x` |
| 0.50 | compact FP8 | 166 | `0.25x` | `[3]` | `0.994x` |
| 0.75 | compact BF16 | 241 | `0.5x` | `[7]` | `1.005x` |
| 0.75 | compact FP8 | 241 | `0.25x` | `[7]` | `1.008x` |
| 1.00 | compact BF16 | 316 | `1.0x` | `[32]` | `1.002x` |
| 1.00 | FP8 / compact FP8 | 316 | `0.5x` | `[3]` | `0.999x / 0.997x` |

32GB observed capacity 使用 600 个 identical multi-image requests、output2、auto pool、prefix caching disabled。关闭 prefix cache 是因为当前 VL prefix-prefill 未实现；否则相同 synthetic token ids 会错误改变本实验的独立请求页占用口径。

| Mode | Pool blocks | Peak running | vs off | Peak allocated | Elapsed | Completed / swap |
|---|---:|---:|---:|---:|---:|---:|
| off BF16 | 249 | 124 | `1.000x` | 28,155.9 MiB | 91.323 s | 600 / 0 |
| compact BF16 | 249 | 248 | `2.000x` | 28,231.1 MiB | 83.510 s | 600 / 0 |
| FP8 | 499 | 249 | `2.008x` | 28,249.4 MiB | 95.602 s | 600 / 0 |
| compact FP8 | 499 | 498 | `4.016x` | 28,399.5 MiB | 111.411 s | 600 / 0 |

容量提升来自每请求 blocks 和每 block dtype bytes 两个正交因素，但组合模式 elapsed 最慢，不能将 `4.016x` max concurrency 写成吞吐提升。near-capacity 模式采用一个 mode 一个进程；四种 pool 在同一 CUDA context 连续创建时曾出现一次 Triton illegal memory access，独立进程和 `CUDA_LAUNCH_BLOCKING=1` 复验成功，资源碎片/生命周期风险仍需后续 hardening。

P6.6 的质量结论是 FAIL，而不是“部分达标”：uniform policy 在多图、视频、mixed 和真实图片上都出现早期 greedy 分叉；单个 COCO 样例不能证明 accuracy drop `<1%`，且当前没有任务级 benchmark score。显存容量和 TPOT 结果证明 physical mechanism 有效，但下一版 pruning 必须使用内容感知/层感知策略并独立测质量，不能继续靠 uniform ratio 调参。

### 5.13 P6.7 Fixed External Eager Baselines

外部 adapter 使用与 Prism 相同 manifest/chat template、BF16、TP1、context 1280、output32、warmup/repeat `1/3`、prefix/MM cache off 和 eager execution。vLLM KV pool 固定为 Prism BF16 16×256 blocks 的 `603,979,776` bytes；SGLang 固定 `max_total_tokens=4096`。汇总器按 manifest hash/case/batch/output 精确匹配，并在 prompt token 数不同或显存测量口径不同时拒绝生成 ratio。

| Framework | Case | Prompt P/E | TPOT P/E | E/P TPOT | Throughput P/E | E/P throughput | Stable prefix |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM 0.24.0 | single-image | 210/210 | 32.366/15.927 ms | `0.492x` | 29.462/55.966 | `1.900x` | 28/32 |
| vLLM 0.24.0 | multi-image | 408/408 | 33.423/16.162 ms | `0.484x` | 26.826/51.966 | `1.937x` | 32/32 exact |
| vLLM 0.24.0 | COCO | 316/316 | 32.273/15.725 ms | `0.487x` | 29.247/54.018 | `1.847x` | 7/32 |
| SGLang 0.5.15 Triton | single-image | 210/210 | 32.366/13.992 ms | `0.432x` | 29.462/66.697 | `2.264x` | 2/32 |
| SGLang 0.5.15 Triton | multi-image | 408/408 | 33.423/13.789 ms | `0.413x` | 26.826/65.174 | `2.430x` | 32/32 exact |
| SGLang 0.5.15 Triton | COCO | 316/316 | 32.273/14.049 ms | `0.435x` | 29.247/64.548 | `2.207x` | 7/32 |

Prism compression-on 的结论没有变化：`visual_compact_fp8 keep=0.5` TPOT 仍约 32-33 ms，vLLM/SGLang 相对 ratio 为 `0.485-0.493x` / `0.414-0.438x`。physical compression 带来容量而非 batch1 decode 加速，且 uniform quality gate FAIL，因此不能声称 Prism 已超过外部框架。

外部环境限制：

- vLLM `0.24.0/ee0da84ab` 的 FlashInfer sampler 错误识别 Blackwell capability，baseline 使用源码支持的 PyTorch native sampler；attention 保持 `FLASH_ATTN`。
- SGLang `v0.5.15/f63458b5` 的 FA3 vision 显式不支持 Blackwell；FA4 在当前 CUTLASS DSL 上编译失败。成功 baseline 使用 text Triton + vision `triton_attn`，不能代表可工作的 FA4 最优性能。
- SGLang memory 是 NVML process-used，Prism/vLLM 是 torch allocator peak，因此只列原始值、不生成 memory ratio。
- video/mixed 的 external prompt tokens 比 Prism 少 2，因为外部 Qwen3-VL processor使用 timestamp video replacement。两行保留 raw record但标记不可比。
- vLLM-Omni clean commit `73bafd64` 的标准 Qwen3-VL 路径就是其固定 vLLM dependency；不重复运行并命名为独立第三框架结果。

这组结果给后续优化明确了目标：Prism 的瓶颈不在已优化的单个 paged kernel，而在每层 eager Python/launch/模型执行路径。要追近外部框架，应把 compression path 接入 CUDA Graph 或更大 compile/fusion region，同时保持 P6.6 quality gate；仅继续微调 KV compaction copy 不会消除约 2x 的 TPOT差距。

### 5.14 P6 Review Conclusion

P6 原阶段 Review full regression 为 `195 passed, 5 skipped in 245.50s`；variable-size TP IPC 后复跑为 `197 passed, 6 skipped in 267.13s`，新增 skip 是显式两卡 integration test。当前单卡环境中的 P1-P6 路径没有已知回归；这个结果不改变性能/质量事实：

1. physical compact + FP8 的主优势是容量，multi-image observed peak running 从 124 提升到 498（`4.016x`），不是 batch1 TPOT。
2. uniform pruning 的质量门禁失败，不能发布 accuracy drop `<1%` claim。
3. Prism eager TPOT 约 32-33 ms，固定 external eager baseline 约 14-16 ms；当前端到端路径约慢 2 倍。
4. CUDA Graph 已证明 Prism off decode 可加速 `1.68x-1.79x`，所以最近的高价值工程方向是 compressed layout 的 graph-safe metadata/control path，而不是继续优化已非主瓶颈的 compaction copy。
5. TP2 目前仍不是已验证能力：fixed 1 MiB control shared memory 已替换为 variable-size Pipe，4,817,396-byte 视觉 payload 双 worker focused test PASS；但当前只有一张 GPU，TP1/TP2 greedy、logits、NCCL、每卡显存与性能均未实测。

因此 P6 的准确表述是“系统测量、关键优化和 variable-size TP control plane 闭环完成，并暴露出质量、framework overhead 与两卡动态验证三个下一阶段问题”，不是“吞吐超过 vLLM/SGLang”。clean commit 之前所有性能数字仍是内部 dirty-worktree evidence。

### 5.15 P6.11 Compressed KV CUDA Graph

P6.11 把 P6.3 已验证的 decode CUDA Graph 扩展到 physical KV compression。实现不修改 paged-attention kernel：capture 时 KV cache 已按 BF16 或 E4M3FN dtype 分配并绑定；每次 replay 前，runner 更新 physical `context_lens`、compact `block_tables` 和 decode append `slot_mapping`。因此 Graph 只消费压缩后的物理布局，不需要在 replay 中重做 pruning decision。

模式边界是显式契约：`off/fp8_kv/visual_compact/visual_compact_fp8` Graph-safe；logical `visual_prune` 的 retained-slot gather 依赖动态 metadata，配置为 Graph 时直接报错。拒绝 silent eager fallback，是为了让 benchmark 的 `execution=cuda_graph` 始终代表真实 replay，而不是标签与执行不一致。

同一 single-image `448x448`、prompt/image tokens `210/196`、output32、16 个 256-token blocks、warmup/repeat `2/5`。下表为 clean commit `9e30e55` formal rerun：

| Compression | Eager median | Graph median | Graph p90/min/max | Speedup | Graph peak allocated | Token/physical KV |
|---|---:|---:|---:|---:|---:|---|
| compact BF16 | 32.3903 ms | 17.6382 ms | 17.6755 / 17.4888 / 17.7253 ms | `1.8364x` | 19,709.5 MiB | exact |
| FP8 dense | 32.4378 ms | 17.6575 ms | 17.6964 / 17.6128 / 55.4299 ms | `1.8371x` | 19,421.5 MiB | exact |
| compact FP8 | 32.4459 ms | 17.5057 ms | 17.5271 / 17.4365 / 17.5467 ms | `1.8535x` | 19,421.5 MiB | exact |

每一行只比较同一种 compression 和 attention backend 的 eager/Graph pair。三组 output SHA256 分别在 pair 内 exact，physical prompt tokens 为 `112/210/112`，active prompt bytes 为 `37,748,736/18,874,368/18,874,368`，Graph 没有改变压缩率或 KV dtype。FP8 Graph raw decode steps 中存在单个 `55.4299 ms` max outlier，median/p90 仍为 `17.6575/17.6964 ms`；报告保留该异常值和完整 p99，不将其删除后重算。

compact FP8 batch matrix 使用同一请求复制形成 offline batch，output32、warmup/repeat `2/5`：

| Batch | Eager median | Graph median | Graph p90/min/max | Speedup | Graph decode tok/s | Physical prompt tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34.0119 ms | 17.5069 ms | 17.5311 / 17.4371 / 17.6271 ms | `1.9428x` | 57.1120 | 112 |
| 2 | 33.6158 ms | 17.7515 ms | 17.7891 / 17.6920 / 18.0370 ms | `1.8937x` | 112.6239 | 224 |
| 4 | 33.5902 ms | 18.3016 ms | 18.3338 / 18.2524 / 19.7760 ms | `1.8354x` | 218.4745 | 448 |
| 8 | 33.6542 ms | 19.1519 ms | 19.2096 / 19.1017 / 20.5820 ms | `1.7572x` | 417.3168 | 896 |

batch1-8 的 eager/Graph output SHA256、physical token 数和 active bytes 均 exact；Graph peak allocated 从 batch1 的 `19,421.5 MiB` 增至 batch8 的 `19,468.9 MiB`。吞吐随 batch 增长，但这是 replicated-request offline decode throughput，不包含在线 arrival、queueing 或调度压力，不能写成 online serving 能力。

correctness 另外覆盖 single-image、multi-image、video、mixed text/image/video、mixed batch=3 -> Graph bucket4，以及 compact FP8 output128；均为同 compression eager/Graph greedy token exact。output128 只做 warmup/repeat `1/1` correctness smoke，不作为稳定性能数字。

边界结论：

1. compressed layout 已不再被迫走约 32-33 ms 的 eager decode，当前 Graph 路径回到约 17.5-19.1 ms，消除了 P6.7 所定位的主要 eager framework/launch overhead。
2. 这不是新的 external benchmark。P6.7 的 vLLM/SGLang 数字使用不同时间点和 eager 对比协议，提交后必须在同一 clean commit、同一 execution 条件下重跑，才能讨论新的 external ratio。
3. Graph 加速 execution，不改善 compression quality。P6.6 uniform pruning quality gate 继续为 FAIL，FP8 与 BF16 长输出之间的既有分叉也继续保留。
4. 初始 correctness/mixed/output128 records 是 commit `ac6e01d`、`git_dirty=true` validation evidence；上面两张正式性能表来自 commit `9e30e55`、`git_dirty=false` formal rerun。output128 尚未作为 clean performance matrix 重跑，只保留 correctness smoke 定位。

Raw evidence：

```text
data/p6_system/p611_physical_graph_batch1_output32_20260714.jsonl
data/p6_system/p611_combo_graph_batch_matrix_output32_20260714.jsonl
data/p6_system/p611_combo_graph_output128_20260714.jsonl
data/p6_system/p611_combo_graph_smoke_20260714.jsonl
data/p6_system/p611_combo_graph_multi_image_20260714.jsonl
data/p6_system/p611_combo_graph_video_20260714.jsonl
data/p6_system/p611_combo_graph_mixed_20260714.jsonl
data/p6_system/p611_clean_physical_graph_batch1_output32_20260714.jsonl
data/p6_system/p611_clean_combo_graph_batch_matrix_output32_20260714.jsonl
```

### 5.16 P6.12-A Runtime Attention Pruning

P6.12-A 将 P5 已存在但未接入 runtime 的 score decision 改为两阶段执行。prefill 前只创建 scorer；选定 decoder layers 在 device 上计算“最后 query 对 visual keys 的 attention probability”，完整 prefill 后才 materialize score、生成 decision，并进入 P6.4 compaction。decode 继续复用 P6.11 CUDA Graph，不在 replay 内重算 score。

第一版选择最后 4 层 attention mean，理由是生成起点的最后 query 与任务文本直接相关，且 P4 trace 已有 `prefill_last_query` 独立语义可验证。拒绝只按 K norm 排序，因为 P5 已将它限定为弱 proxy；也拒绝每层写 CPU record，因为这会在 prefill 中引入同步。当前没有采用外部 pruning 实现，设计来自本项目 trace/compaction contract 和 first-principles relevance ranking。

单元 reference 使用 q `[8,4,2]`、k `[8,2,2]` 的 GQA，runtime/reference score `[5]` mean/std 都为 `1.203456e-01/2.894921e-02`，max diff `0`。真实 Qwen3-VL 记录的 score layers 为 `[32,33,34,35]`，decision 同时保存 source、min/max/mean，便于后续审计 layer/strategy 变化。

keep=0.5 quality preflight：

| Workload | Output | Uniform prefix | Attention prefix | Logical/physical | Active BF16 bytes |
|---|---:|---:|---:|---:|---:|
| COCO `000000039769` | 32 | 3 | 21 | 316 / 166 | 37,748,736 |
| multi-image `2x448` | 128 | 6 | 7 | 408 / 212 | 37,748,736 |
| video `4x448` | 128 | 14 | 14 | 422 / 226 | 37,748,736 |

attention ranking 没有改变 physical compression ratio。COCO 改善显著，但 multi-image 只多保持 1 token，video 无改善；3 个 workload 也没有任务级 accuracy，因此 quality gate 仍是 FAIL。不能把 COCO 的 `3 -> 21` 外推成“内容感知 pruning 已解决质量问题”。

机制 smoke 覆盖：

- single-image compact BF16 eager/Graph output8 exact。
- single-image compact FP8 eager/Graph output8 exact，physical prompt `112`。
- mixed batch=3 Graph 中 text row 保持 dense，image/video 分别 compact 到 `112/226`。

尝试过 coverage-aware greedy MMR，weight `0.25`，希望缓解 pure top-k 的空间聚集。实测 COCO/multi/video prefix 为 `7/6/14`，不如 pure attention 的 `21/7/14`；Python greedy loop 还使单次观察 prefill 增至约 `236-390 ms`。该候选同时损害质量和 TTFT，代码已删除，raw record 只作为 rejected ablation evidence。

当前 scorer 性能限制：每个选定层额外执行一次 last-query × full-context score matmul；TP finalize 还需要按 sequence all-reduce。P6.12-A 的 `warmup=1/repeat=1` 只用于质量/smoke，不满足稳定性能 claim。下一步若质量策略通过，必须用 warmup/repeat `2/5` 单独测 scorer TTFT overhead、Graph TPOT、memory 和 TP communication。

Raw evidence：

```text
data/p6_system/p612_attention_graph_smoke_20260714.jsonl
data/p6_system/p612_attention_combo_graph_smoke_20260714.jsonl
data/p6_system/p612_attention_mixed_graph_smoke_20260714.jsonl
data/p6_system/p612_coco_uniform_quality_20260714.jsonl
data/p6_system/p612_coco_attention_quality_20260714.jsonl
data/p6_system/p612_multi_image_uniform_quality_20260714.jsonl
data/p6_system/p612_multi_image_attention_quality_20260714.jsonl
data/p6_system/p612_video_uniform_quality_20260714.jsonl
data/p6_system/p612_video_attention_quality_20260714.jsonl
data/p6_system/p612_*_attention_mmr025_quality_20260714.jsonl
data/p6_system/p612_clean_coco_uniform_quality_20260714.jsonl
data/p6_system/p612_clean_coco_attention_quality_20260714.jsonl
data/p6_system/p612_clean_multi_image_uniform_quality_20260714.jsonl
data/p6_system/p612_clean_multi_image_attention_quality_20260714.jsonl
data/p6_system/p612_clean_video_uniform_quality_20260714.jsonl
data/p6_system/p612_clean_video_attention_quality_20260714.jsonl
```

初始 P6.12 smoke/MMR records 为 commit `39802be`、`git_dirty=true` validation。COCO/multi-image/video 的 off/uniform/attention 9 条关键 quality records 已在 commit `c07fa34`、`git_dirty=false` 上 formal rerun，表中 prefix 与 physical-token 数据完全复现。P6.12-A engineering/correctness PASS、quality FAIL；外部 vLLM/SGLang ratio 没有更新，scorer 稳定性能矩阵也尚未执行。

### 5.17 P6.12-B Per-span Budget Ablation

clean global-attention records 显示，multi-image 的两个 196-token image span
分别保留 `124/72`，video 的两个 196-token span 分别保留
`109/87`。这证明 global top-k 会在 span 之间形成不均匀预算，但它本身不能
证明不均匀就是质量分叉的原因。

为检验该假设，临时候选先按 span token capacity 分配 largest-remainder
quota，再在 span 内做 attention top-k。它保持总 keep 数和 physical KV 不变，
并将两个双 span workload 都改为 `98/98`。候选没有采用外部实现，
来源是上述 clean decision 的 first-principles budget ablation。

| Workload | Global prefix | Per-span prefix | Global/per-span physical | 结论 |
|---|---:|---:|---:|---|
| COCO `000000039769` | 21 | 21 | `166/166` | 32-token exact，无改善 |
| multi-image `2x448` | 7 | 7 | `212/212` | 同一位置首次分叉，无改善 |
| video `4x448` | 14 | 14 | `226/226` | 128-token exact，无改善 |

该矩阵为 keep `0.5`、last 4 layers、greedy、`warmup=1/repeat=1`，
仅用于 quality preflight。等额 quota 没有使任何 stable prefix 超过 global
attention，所以候选代码已删除。当前支持的 `attention` 只增加
`kept_visual_tokens_by_span` 审计字段，不改变 selection 语义。

Raw evidence：

```text
data/p6_system/p612b_coco_attention_span_quality_20260715.jsonl
data/p6_system/p612b_multi_image_attention_span_quality_20260715.jsonl
data/p6_system/p612b_video_attention_span_quality_20260715.jsonl
```

这个结果拒绝了“只要平衡 span 预算就能改善质量”的假设。P6.12-B
下一步应将候选与更大固定数据集一起设计，考察 grid coverage、跨 query/layer
聚合或非等额动态预算；当前 quality 继续 FAIL，也没有新的性能收益 claim。

### 5.18 P6.12-B Seven-image Fidelity Matrix

固定真实集从单个 COCO 样例扩展为 7 张 COCO val2017 图片，
按 `4+3` requests 分成两个 batch case。每个资产固定官方 URL、SHA256
和尺寸。dataset summary 以 schema-v4+ physical KV、完整 candidate case
coverage 和相同 model/execution config 汇总 stable-prefix、exact rate、KV ratio
和 span starvation；schema-v5 reference-task evidence 见 5.19。

quality preflight 使用 RTX 5090、bf16、CUDA Graph、greedy output32、keep=0.5、
min keep=32、last 4 layers、warmup/repeat `1/1`。mixed-VL batch 显式关闭
prefix caching，实际开关已写入 benchmark model metadata。

| Strategy | Exact requests | Prefix micro | Median | Min | Physical KV | Active bytes | Zero spans |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention last4 | `3/7` | `0.696` | `0.875` | `0.219` | `0.535x` | `0.538x` | `0/7` |
| uniform | `0/7` | `0.304` | `0.188` | `0.094` | `0.535x` | `0.538x` | `0/7` |

单请求 prefix 为：

- attention: `[7,11,32,28,14,32,32]`。
- uniform: `[3,6,6,10,18,19,6]`。

这是 attention 相对 uniform 的 dataset-level greedy fidelity 改善，而不是 task
accuracy PASS。该段记录的是 reference 接入前的 fidelity-only preflight；随后
schema-v5 多参考 caption 门禁已完成，结果与限制见 5.19。

这些记录为 commit `225b289`、`git_dirty=true` validation，且 `1/1` 只用于
quality preflight，不用于 TTFT/TPOT/throughput claim。

Raw evidence：

```text
data/p6_system/p612b_coco_fidelity_batch_a_attention_20260715.jsonl
data/p6_system/p612b_coco_fidelity_batch_b_attention_20260715.jsonl
data/p6_system/p612b_coco_fidelity_batch_a_uniform_20260715.jsonl
data/p6_system/p612b_coco_fidelity_batch_b_uniform_20260715.jsonl
data/p6_system/p612b_coco_fidelity_strategy_summary_20260715.json
data/p6_system/p612b_coco_fidelity_strategy_summary_20260715.md
```

### 5.19 P6.12-B Multi-reference Caption Quality Gate

schema-v5 在 correctness 中保存每请求 decoded text/hash，在 workload 中保存
COCO reference source 与 task identity；output decoding 明确发生在计时结束后。
7 张固定 COCO val2017 图片各绑定 5 条 caption。token-F1 与 token-level
ROUGE-L F1 分别在 5 条 reference 中取最高分，再对 7 个 requests 做 macro；
candidate 相对 off baseline 的两项绝对下降都必须 `<=0.01`。

| Strategy | Token-F1 B/C | Drop | ROUGE-L B/C | Drop | Gate |
|---|---:|---:|---:|---:|:---:|
| attention last4 | `0.321635/0.315285` | `0.006351` | `0.289116/0.276703` | `0.012413` | FAIL |
| uniform | `0.321635/0.315486` | `0.006150` | `0.289116/0.252751` | `0.036365` | FAIL |

attention 的 token-F1 drop 在阈值内，但 ROUGE-L drop 超出 `0.002413`，所以
task gate FAIL。uniform 的 ROUGE-L drop 为 `0.036365`，退化更明显。结合
5.18，attention 在 stable-prefix 和 ROUGE-L retention 上明显优于 uniform；但
uniform candidate token-F1 略高于 attention，不能声称 attention 在所有 task
metric 上支配 uniform。

这是相对 off baseline 的 lexical quality preflight，不是 COCO 官方 CIDEr/SPICE。
生成只到 32 tokens，而 prompt 多要求 detailed description、COCO references 较短，
所以约 `0.3` 的绝对分数不应用作完整 caption 能力结论。固定 7-image suite 也不足
以发布 `accuracy drop <1%` claim。

本轮使用 RTX 5090、bf16、CUDA Graph、keep=0.5、min keep=32、last4、
prefix cache off、warmup/repeat `1/1`。记录属于 commit `9e5db53`、
`git_dirty=true` quality validation；output decoding 不进入 E2E，但该矩阵仍不
支持 TTFT/TPOT/throughput claim。focused tests 为 `54 passed in 3.84s`，
受影响回归为 `100 passed in 7.50s`。

Raw evidence：

```text
data/p6_system/p612b_task_quality_batch_a_attention_20260715.jsonl
data/p6_system/p612b_task_quality_batch_b_attention_20260715.jsonl
data/p6_system/p612b_task_quality_batch_a_uniform_20260715.jsonl
data/p6_system/p612b_task_quality_batch_b_uniform_20260715.jsonl
data/p6_system/p612b_task_quality_strategy_summary_20260715.json
data/p6_system/p612b_task_quality_strategy_summary_20260715.md
```

### 5.20 P6.12-C Final-layer Attention Quality Pass

P6.12-C 保持 last-query attention、global top-k、keep ratio `0.5`、min keep
`32`、physical compaction 和 CUDA Graph 不变，只消融聚合的 decoder layer
数量。batch A 的 last1/last4/last8 结果表明，增加层数没有带来 task-quality
改善：

| Strategy | Prefix micro | Token-F1 drop | ROUGE-L drop | Batch-A gate |
|---|---:|---:|---:|:---:|
| attention last1 | `0.531` | `0.008275` | `0.008275` | PASS |
| attention last4 | `0.609` | `0.012899` | `0.012899` | FAIL |
| attention last8 | `0.609` | `0.012899` | `0.012899` | FAIL |

随后在 clean commit `a7588d3` 上对两个 COCO batch 重新成对执行
`off_graph,visual_compact_graph`，baseline/candidate 都为 `git_dirty=false`。
last1 的 7-image/35-reference 正式 preflight 为：

| Strategy | Exact requests | Prefix micro/min | Physical KV | Active bytes | Token-F1 B/C | ROUGE-L B/C | Gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| attention last1 | `3/7` | `0.652/0.094` | `0.535x` | `0.538x` | `0.321635/0.318347` | `0.289116/0.285406` | PASS |

token-F1 drop 为 `0.003288`，ROUGE-L drop 为 `0.003710`，均低于
`0.01`。这满足项目定义的首个 reference-task preflight gate，同时保持与 last4
相同的物理压缩率。last1 的 prefix micro 低于 last4 (`0.652 < 0.696`) 且最差
样本 prefix 只有 3/32，因此结论是“任务词法门禁通过”，不是所有输出都更接近
off，也不是 COCO 官方 CIDEr/SPICE 或 `accuracy drop <1%`。

当前实现据此把 attention scorer 默认层数从 4 改为 1；decision 继续记录
`score_layers=[35]`，benchmark mode 继续记录
`visual_pruning_attention_last_n_layers=1`，显式 override 与历史 last4 records
保持兼容。该结果来自项目内部 layer ablation，不能归因为 PoRe、SnapKV 或
TokenCarve 的复现。

默认层数切换提交 `e51c16d` 后，不传 last-N 参数、显式选择 attention 的两个
COCO batch 继续记录 `attention:last1` 与 `score_layers=[35]`，两项 task drop
完全复现为 `0.003288/0.003710`，baseline/candidate 均为 clean。默认切换没有
改变 decision schema，也没有破坏显式 last4 override。

multi-image/video/mixed clean smoke：

| Workload | Output | Stable prefixes | Physical token ratio | Active bytes | Span starvation |
|---|---:|---|---:|---:|:---:|
| multi-image 2x448 | 128 | `[7]` | `0.520x` | `0.500x` | no |
| video 4x448 | 128 | `[14]` | `0.536x` | `0.500x` | no |
| mixed text/image/video | 32 | `[32,28,14]` | `0.539x` | `0.750x` | no |

这些 smoke 与历史 last4 的 multi-image/video prefix `7/14` 一致，只证明默认层数
切换没有引入执行或多 span 回归；它们没有 reference task evidence，不能扩展
7-image caption gate 的质量结论。

同一 clean commit 的 COCO batch A、batch4、output32、CUDA Graph、
`warmup=2/repeat=5` 稳定矩阵：

| Mode | Prefill median | Decode-step median | Decode tok/s | Engine output tok/s | E2E median | Physical tokens | Active bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| off_graph | `221.874 ms` | `18.945 ms` | `211.008` | `158.048` | `993.238 ms` | `988` | `264,241,152` |
| attention last1 compact Graph | `224.179 ms` | `18.553 ms` | `215.571` | `160.087` | `988.486 ms` | `530` | `150,994,944` |

last1 相对 off 的 prefill ratio 为 `1.010x`，decode-step speedup `1.021x`，
decode throughput `1.022x`，engine output throughput `1.013x`，E2E speedup
`1.005x`；physical token/active-byte ratio 为 `0.536x/0.571x`。固定 KV pool
导致 peak allocated 基本不变，因此 active bytes/page reclamation 才是该矩阵的
容量证据。

显式 last4 同条件 prefill/decode median 为 `223.938/18.544 ms`；last1/last4
差异小于 `0.2%`，不能声称 last1 减少 scorer 成本。last1 的默认选择由 task
quality PASS 驱动，decode 收益来自两者共有的 physical compaction 与更短 context。

最终 full regression 生成 JUnit evidence，结果为
`238 passed, 6 skipped in 232.90s`。skip 继续包含单卡环境无法执行的 TP2
integration；没有新增回归失败。

Raw evidence：

```text
data/p6_system/p612c_batch_a_attention_last1_20260716.jsonl
data/p6_system/p612c_batch_a_attention_last8_20260716.jsonl
data/p6_system/p612c_clean_task_quality_batch_a_attention_last1_20260716.jsonl
data/p6_system/p612c_clean_task_quality_batch_b_attention_last1_20260716.jsonl
data/p6_system/p612c_clean_attention_last1_quality_summary_20260716.json
data/p6_system/p612c_clean_attention_last1_quality_summary_20260716.md
data/p6_system/p612c_default_clean_task_quality_batch_a_20260716.jsonl
data/p6_system/p612c_default_clean_task_quality_batch_b_20260716.jsonl
data/p6_system/p612c_default_clean_quality_summary_20260716.json
data/p6_system/p612c_default_clean_multi_image_output128_20260716.jsonl
data/p6_system/p612c_default_clean_video_output128_20260716.jsonl
data/p6_system/p612c_default_clean_mixed_output32_20260716.jsonl
data/p6_system/p612c_default_clean_multimodal_fidelity_summary_20260716.json
data/p6_system/p612c_default_clean_performance_batch_a_output32_20260716.jsonl
data/p6_system/p612c_clean_performance_batch_a_attention_last4_output32_20260716.jsonl
data/p6_system/p612c_full_regression_20260716.xml
```

## 6. P7 单机性能与外部对标

### 6.1 P7.0 P6.12 Freeze

P6.12 content-aware BF16 主线冻结在 commit `c970c61`，annotated tag
`p6.12-content-aware-kv` 已推送。允许/禁止 claim 独立记录在
`docs/CLAIMS.md`；性能问题的调查过程记录在 `docs/issues/`。

### 6.2 P7.1 External Benchmark Protocol v2

benchmark harness commit 为 clean `b17f933`。双方固定同一 RTX 5090 GPU UUID、
Qwen3-VL config SHA256、BF16、TP1、max model len 1280、max batched tokens 2048、
block 256、KV pool `603,979,776` bytes、prefix cache off、output32、temperature 0、
ignore EOS、warmup/repeat `2/5`。

vLLM 固定 `0.24.0/ee0da84ab`、Torch `2.11.0+cu130`、`FLASH_ATTN` 和 PyTorch
native sampler。Prism 使用 Torch `2.6.0a0+nv25.01` 和自实现 paged decode。
runtime/framework/backend版本属于被比较系统，可以不同，但已写入每条 record。

schema-v2 对 model hash、GPU UUID、prompt token、KV pool、block、sampling、
warmup/repeat、计时 scope、effective execution 和 clean state逐项校验。两组共
20 个 comparison rows 全部 `performance_comparable=true`，没有手工放行 cell。

### 6.3 Diagnostic Matched Eager

`diagnostic_matched` 中双方关闭 CUDA Graph；vLLM 额外关闭 chunked prefill 和
async scheduling。下表只列 dense Prism off；content-aware eager TPOT 与 off
差异小于约 `0.3%`，不能消除 eager execution gap。

| Workload | Prompt | Prism eager TPOT | vLLM eager TPOT | Prism/vLLM | Prism/vLLM E2E output tok/s |
|---|---:|---:|---:|---:|---:|
| single image | 210 | `32.524 ms` | `17.012 ms` | `1.912x` | `29.218/53.691` |
| two images | 408 | `32.403 ms` | `16.946 ms` | `1.912x` | `28.940/53.077` |
| real COCO | 316 | `32.363 ms` | `16.810 ms` | `1.925x` | `29.272/54.625` |
| fidelity batch A (4) | 988 | `33.341 ms` | `17.363 ms` | `1.920x` | `92.121/188.758` |
| fidelity batch B (3) | 899 | `33.860 ms` | `17.172 ms` | `1.972x` | `69.102/137.561` |

这复现了 P6.7 的约 2 倍 eager 差距，并证明差距不是旧 dirty commit 或 prompt
token 不一致造成。

### 6.4 Best-stable CUDA Graph

Prism 使用 `off_graph/visual_compact_graph`；vLLM effective config 逐条记录为
`VLLM_COMPILE + FULL_AND_PIECEWISE`，chunked prefill/async scheduling enabled。

| Workload | Prism off | Prism compact | vLLM | Off/vLLM | Compact/vLLM |
|---|---:|---:|---:|---:|---:|
| single image | `17.932` | `17.671` | `10.098 ms` | `1.776x` | `1.750x` |
| two images | `18.501` | `17.959` | `10.119 ms` | `1.828x` | `1.775x` |
| real COCO | `18.234` | `17.822` | `10.105 ms` | `1.805x` | `1.764x` |
| fidelity batch A (4) | `18.966` | `18.574` | `10.822 ms` | `1.752x` | `1.716x` |
| fidelity batch B (3) | `18.536` | `18.120` | `10.970 ms` | `1.690x` | `1.652x` |

Prism 的 Graph 相对同 compression eager约快 `1.75x-1.86x`，但 vLLM 也从 eager约
`16.8-17.4 ms` 降至 `10.1-11.0 ms`。因此 Graph缩小但没有消除 external gap；
当前不能声称 Prism 的 raw latency/throughput 超过 vLLM。

content-aware compact 相对 Prism off Graph 的 TPOT改善约 `1.5%-3.0%`。这与
P6.12 的结论一致：压缩的第一收益仍是物理 KV/page容量，不是短 context 的
大幅 latency。

### 6.5 Physical KV 与固定 pool 显存

| Workload | Physical tokens off/compact | Active prompt bytes off/compact |
|---|---:|---:|
| single image | `210/112` | `37,748,736/37,748,736` |
| two images | `408/212` | `75,497,472/37,748,736` |
| real COCO | `316/166` | `75,497,472/37,748,736` |
| fidelity A | `988/530` | `264,241,152/150,994,944` |
| fidelity B | `899/479` | `226,492,416/113,246,208` |

single-image 的 retained tokens 仍落在同一个 256-token page，因而没有 active
byte下降。固定 pool 下 Prism peak allocated 约 `19.7 GiB`，vLLM 约
`17.7-17.9 GiB`；active page 回收不会缩小预分配 pool 或模型权重峰值。

### 6.6 Graph Residual Gap Attribution

clean single-image semantic CUDA profile（只用于归因）将 Prism decode 分解为：

| Region | Off Graph | Compact Graph |
|---|---:|---:|
| Graph replay | `13.394 ms` | `13.124 ms` |
| logits | `4.068 ms` | `4.068 ms` |
| Graph input copy | `0.129 ms` | `0.130 ms` |
| sampler GPU | `0.175 ms` | `0.174 ms` |

Graph replay 约占 TPOT四分之三，Graph 外 logits 约占 23%；压缩只将 replay
减少约 `0.27 ms`。所以下一轮 profiling 应优先比较 Graph 内 decoder kernels
和 logits，不再继续优化已非关键路径的 compaction copy。

### 6.7 Variance Boundary

TPOT分布稳定，但 offline vision prefill/TTFT 出现双峰。single-image
`warmup=5/repeat=15` 中 off Graph TTFT median/p90/min/max 为
`131.468/139.074/50.040/141.531 ms`，compact Graph 为
`58.037/132.139/51.271/136.011 ms`；对应 TPOT p90 仅
`17.957/17.709 ms`。semantic profile 将主要波动定位到 eager vision prefill，
但当前环境不能锁 GPU clocks，根因尚未充分证明。

因此本节 headline 使用稳定 TPOT。E2E/TTFT原始分布保留，但不把 compact/off
E2E中位数差异归因于压缩；详见
`docs/issues/P7-005-TTFT_VISION_BIMODALITY.md`。

Raw evidence：

```text
data/p7_external/prism_*_formal_b17f933.jsonl
data/p7_external/vllm_*_diagnostic_matched_formal_b17f933.json
data/p7_external/vllm_*_best_stable_formal_b17f933.json
data/p7_external/p7_diagnostic_matched_summary_b17f933.{json,md}
data/p7_external/p7_best_stable_summary_b17f933.{json,md}
data/p7_external/prism_single_image_graph_stability_b17f933.jsonl
data/p7_external/prism_single_image_graph_semantic_profile_b17f933.jsonl
data/p7_external/p71_full_regression_20260716.xml
```

### 6.8 Verification

- focused schema/summary regression：`42 passed in 3.79s`。
- 两条 summary 从正式 raw records 重生成后与保存结果逐字节一致；共 20 rows，
  `20 performance_comparable / 0 non-comparable`。
- full regression JUnit：`tests=246`、`failures=0`、`errors=0`、`skipped=6`、
  `time=232.301s`，即 `240 passed, 6 skipped`。

### 6.9 P7.4-A Trace-driven Logits Projection

P7.1 的 semantic profile 已把 Graph 外 logits 定位为 `4.068 ms/step`，但没有
解释 kernel。P7.4 在同一 single-image/output32 workload 上启用 Nsight Systems
node-level CUDA Graph trace，并与 vLLM best-stable capture 对照。旧 Prism capture
中最显著的 Graph 外工作是：

- BF16→FP32 `direct_copy`：全 capture `96.141 ms`。
- FP32 vocab GEMV：32 次合计 `48.282 ms`，median约 `1.509 ms`。
- `compute_logits` region：median `4.068 ms`。

根因是旧 `Qwen3VLForCausalLM.compute_logits()` 在每个 prefill/decode step 调用
`F.linear(hidden_states.float(), lm_head.weight.float())`。Qwen3-VL-8B 的 lm-head
为 `151,936 × 4,096`；每步把整张 BF16 权重临时转成 FP32既产生大显存流量，也
产生约 `2.3 GiB` transient allocation。vLLM capture 没有对应整权重转换路径。

commit `a33e7ed` 将默认改为模型原生精度 lm-head，保留显式
`logits_precision=fp32` 只用于历史复现；benchmark record同步记录该字段，vLLM
harness增加 CUDA Profiler API capture range。优化后 clean `cc070b3` trace：

| Region / metric | FP32 historical | Model precision | 变化 |
|---|---:|---:|---:|
| logits CUDA median | `4.068 ms` | `0.762 ms` | `5.34x` faster |
| logits kernels/range median | `4` | `1` | `-3` |
| Graph replay CUDA median | `13.359 ms` | `12.927 ms` | 不归因，属于运行间波动/后续目标 |
| full decode TPOT, single off Graph | `17.887 ms` | `14.151 ms` | `1.264x` faster |
| peak allocated, single off Graph | `19,708.6 MiB` | `17,391.5 MiB` | `-2,317.2 MiB` |

#### 6.9.1 Clean single-variable matrix

所有行来自同一 clean `a33e7ed`、相同 workload/mode、`warmup=2/repeat=5`；只改变
`logits_precision`。

| Workload | Mode | FP32 TPOT | Model TPOT | Speedup | Peak saved |
|---|---|---:|---:|---:|---:|
| single image | off / compact | `17.887 / 17.636` | `14.151 / 14.059` | `1.264x / 1.254x` | `2,317 / 2,317 MiB` |
| two images | off / compact | `18.483 / 17.943` | `14.445 / 14.192` | `1.280x / 1.264x` | `2,287 / 2,287 MiB` |
| real COCO | off / compact | `18.205 / 17.812` | `14.286 / 14.129` | `1.274x / 1.261x` | `2,309 / 2,308 MiB` |
| fidelity A (4) | off / compact | `18.946 / 18.563` | `15.578 / 15.184` | `1.216x / 1.223x` | `2,230 / 2,230 MiB` |
| fidelity B (3) | off / compact | `18.529 / 18.119` | `15.158 / 14.738` | `1.222x / 1.229x` | `2,239 / 2,239 MiB` |

真实 COCO 的 model-precision greedy token 不要求与历史 FP32 path exact；正确性参考
是 HF 模型精度与固定 reference task。single-image、multi-image、video 的 32-token
teacher-forced logits和 PPL相对 HF均为 `max diff=0 / mean diff=0 / ppl diff=0`。
7-image content-aware gate 同样 PASS：

| Metric | Dense | Compact | Delta | Gate |
|---|---:|---:|---:|:---:|
| token-F1 macro | `0.318842` | `0.314482` | `-0.004360` | PASS |
| ROUGE-L macro | `0.285863` | `0.289953` | `+0.004090` | PASS |
| physical tokens | - | - | `0.535x` | PASS |
| active prompt bytes | - | - | `0.538x` | PASS |

#### 6.9.2 Updated vLLM best-stable comparison

vLLM `0.24.0/ee0da84ab` 在同一 clean harness、GPU、model hash、KV pool、block、
sampling 和 `warmup/repeat=2/5` 下重跑；10/10 rows通过 comparability gates。

| Workload | Prism off | Prism compact | vLLM | Compact/vLLM | Peak compact/vLLM |
|---|---:|---:|---:|---:|---:|
| single image | `14.151` | `14.059` | `10.098 ms` | `1.392x` | `17,392.8 / 17,741.2 MiB` |
| two images | `14.445` | `14.192` | `10.119 ms` | `1.402x` | `17,428.7 / 17,794.3 MiB` |
| real COCO | `14.286` | `14.129` | `10.106 ms` | `1.398x` | `17,404.3 / 17,769.3 MiB` |
| fidelity A (4) | `15.578` | `15.184` | `10.828 ms` | `1.402x` | `17,504.7 / 17,934.4 MiB` |
| fidelity B (3) | `15.158` | `14.738` | `10.970 ms` | `1.343x` | `17,492.4 / 17,913.8 MiB` |

优化将 compact Prism 的 TPOT差距从 P7.1 的 `1.65x-1.78x` 缩小到
`1.34x-1.40x`，并让 torch allocator peak低于同条件 vLLM；但 Prism仍未反超
TPOT，E2E throughput也仍受 prefill/TTFT影响，因此禁止写“性能超过 vLLM”。

首次 full regression 暴露两个旧 mixed-batch 测试把 batch1 GEMV 与 batch4 GEMM
要求 token exact。model precision 与 HF逐值一致，但低精度跨 shape 的视频首 token
可在低 margin下分叉。最终合同要求同一 mixed shape重复 exact、image/multi-image
跨 shape 长前缀、HF logits/PPL exact和 reference-quality PASS；clean `cc070b3`
JUnit为 `247 tests / 0 failures / 0 errors / 6 skipped / 264.664s`，即
`241 passed, 6 skipped`。

Raw evidence：

```text
data/p7_external/p74_prism_*_{fp32,model}_formal_a33e7ed.jsonl
data/p7_external/p74_model_quality_summary_a33e7ed.{json,md}
data/p7_external/p74_vllm_*_best_stable_formal_a33e7ed.json
data/p7_external/p74_best_stable_summary_a33e7ed.{json,md}
data/p7_external/prism_single_image_off_graph_nsys_66e6f9f.{nsys-rep,sqlite}
data/p7_external/p74_prism_single_image_model_nsys_cc070b3.{nsys-rep,sqlite}
data/p7_external/p74_prism_single_image_model_nsys_summary_cc070b3.json
data/p7_external/p74_full_regression_cc070b3.xml
```

### 6.10 P7.3 Online Arrival、Continuous Batching 与 SLO Goodput

P7.1/P7.4 的 offline matrix在开始测量前一次性提交固定 batch，不能回答 queue、
arrival或 SLO goodput。P7.3 新增 wall-clock arrival loop；请求可在其他 batch执行期间
到达，并在下一调度点进入 immutable `BatchPlan`。每个 request记录 intended arrival、
first scheduled、first token、逐 token和 terminal时间。

实现过程中确认旧 chunked prefill只有 scheduler外壳：第二个 chunk的 Q<K没有 paged
attention。修复后采用 correctness-first paged gather + bottom-right causal SDPA，并将
视觉 payload整体作为 atomic region。详细根因见
`docs/issues/P7-007-CHUNKED-PREFILL-STATE.md`。

clean `e7796e9` formal matrix：

| Workload | Mode / rate | Requests | Queue p99 | TTFT p99 | TPOT p99 | Req/s | Goodput/s | Peak active |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text short | off Graph / 20 req/s | 16 | `13.434` | `41.502` | `31.096` | `18.245` | `18.245` | `5` |
| single image | off Graph / 4 req/s | 8 | `10.579` | `146.157` | `18.632` | `4.025` | `4.025` | `1` |
| single image | compact Graph / 4 req/s | 8 | `8.094` | `91.992` | `14.383` | `4.197` | `4.197` | `1` |
| mixed text/image/video | off Graph / 4 req/s | 6 | `13.860` | `180.475` | `14.574` | `3.911` | `3.911` | `1` |
| mixed text/image/video | compact Graph / 4 req/s | 6 | `16.376` | `235.634` | `14.216` | `3.767` | `3.767` | `1` |
| mixed text/image/video | off Graph / 10 req/s | 6 | `104.806` | `244.236` | `81.708` | `6.947` | `6.947` | `5` |
| mixed text/image/video | compact Graph / 10 req/s | 6 | `58.482` | `206.415` | `59.567` | `7.327` | `7.327` | `4` |
| text 301-token chunked | off Graph / 4 req/s | 4 | `0.894` | `105.263` | `13.594` | `4.465` | `4.465` | `1` |
| image+text 646-token chunked | off Graph / 2 req/s | 2 | `15.322` | `151.996` | `15.555` | `2.863` | `2.863` | `1` |

SLO不是跨 workload统一后倒推：text-short为 TTFT/TPOT `500/50 ms`，single-image为
`1000/50 ms`，mixed与长输入为 `2000/100 ms`（text-long TTFT为 `1000 ms`）。
9/9 cells的 completed request均满足各自预先声明的 SLO，goodput fraction为 `1.0`。

机制/correctness证据：

- text 301 tokens固定形成 `128/128/45` 三个 prefill chunks；image+text 646 tokens
  固定形成 `512/134`，两者相对 single-prefill输出 exact。
- 10 req/s mixed-VL形成 batch size `1..5`，证明不是 offline replicated batch。
- single-image和 mixed rate10 的 off/compact逐 request 8-token输出 exact。
- concurrent text prefix hit中第二条复用 256-token full block且输出 exact；VL prefix
  hash显式禁用，因为 token id不包含像素语义。
- admission reject、cancel、swap/recompute preemption、queue/KV peaks均有 deterministic
  contract tests；正式 matrix未触发 preemption，不能据此声称 swap性能。

每个 cell是一次含多个 arrival/request的 formal run，没有跨进程 repeat，因此表中
off/compact差异只作为当前 online observation，不形成“compact speedup”结论。当前也
没有相同 arrival/SLO的 vLLM online record，禁止把这些 goodput数字写成外部反超。

Raw evidence：

```text
data/p7_online/p73_*_formal_e7796e9.json
data/p7_online/p73_online_matrix_summary_e7796e9.{json,md}
data/p7_online/p73_full_regression_e7796e9.xml
```

### 6.11 P7.4-B CUDA Graph Replay Anatomy 与 Bucket Coverage

P7.4-B 在 model-precision logits 已闭环后重新采集 single-image/output32 的
node-level Systems trace。capture 使用 clean `0fdd4a6`、CUDA Profiler API range
和 `--cuda-graph-trace=node`，只分析 31 个 measured decode replay；schema-v2
analyzer 同时计算 kernel busy union、GPU activity span、CPU/GPU busy overlap、CPU
range 返回后的 GPU tail，以及每个 NVTX range 的直接 GPU activity。

Graph replay 的每步中位数分解如下：

| Kernel category | Kernels/step | Kernel busy/step | Replay fraction |
|---|---:|---:|---:|
| linear/GEMV | `253` | `9.123 ms` | `70.55%` |
| paged decode attention | `36` | `1.693 ms` | `13.17%` |
| elementwise | `1,157` | `1.165 ms` | `9.02%` |
| copy/cast | `295` | `0.560 ms` | `4.33%` |
| reduction | `145` | `0.233 ms` | `1.80%` |
| layout/index + KV store + trigonometric | `114` | `0.147 ms` | `1.14%` |

八类合计 `2,000 kernels/step`，分类 fraction 精确闭合。replay kernel busy
median/p90 为 `12.921/12.933 ms`，GPU busy union median 为 `12.922 ms`；整个
engine decode 的 kernel busy median 为 `13.690 ms`，因此 Graph 外直接 kernel
busy差为约 `0.769 ms`。这与优化后 logits `0.762 ms`一致，说明 logits 已退出首要
优化目标，下一候选首先是占 replay `70.55%` 的 linear/GEMV 路径。

#### 6.11.1 CPU/GPU timeline 解释

`graph.replay()` CPU range median 为 `1.899 ms`，与 GPU busy overlap只有
`0.030 ms`（`1.62%`）；CPU range返回后 GPU仍执行 median `13.089 ms`。这是异步
提交的预期行为，不代表 Graph只执行了约 1.9 ms。GPU activity span median
`14.957 ms` 比 busy union多 `2.035 ms`，但 node tracing本身带有 instrumentation，
这段差值不能当作 occupancy或可消除 idle 百分比。

Graph 外各 range 的直接 activity进一步排除了重复计时：prepare inputs、prepare
sample inputs和 Graph input copy的直接 GPU busy分别只有约 `0.002/0.000/0.006 ms`；
logits为 `0.762 ms`。sampler CPU range虽然为 `13.790 ms`，其直接 GPU busy只有
`0.007 ms`、semantic CUDA-event elapsed为 `0.187 ms`。该 CPU时间暴露的是同一
stream上等待前序 Graph/logits完成的同步，不是可与 replay相加的独立 sampler成本。
完整解释见 `docs/issues/P7-008-CUDAGRAPH-TIMELINE-ACCOUNTING.md`。

#### 6.11.2 固定 capture ceiling 的 bucket/padding coverage

commit `00b1012` 将实际 offline batch与 `max_num_seqs` capture ceiling解耦。固定
`max_num_seqs=8`、captured buckets `[1,2,4,8]`，batch `1..8` 的 clean matrix为：

| Actual batch | Selected bucket | Padding rows |
|---:|---:|---:|
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 4 | 1 |
| 4 | 4 | 0 |
| 5 | 8 | 3 |
| 6 | 8 | 2 |
| 7 | 8 | 1 |
| 8 | 8 | 0 |

8/8 cells在 `warmup=2/repeat=5` 中重复稳定；复制的 single-image request在所有
batch和 padding row之间生成完全相同的 token ids，证明 bucket选择、padding隔离和
batch 1-8 coverage正确。每个 cell仍是一次独立进程级 run，因此 cell间 TPOT不能
用来声称 padding带来加速或减速；这也不是 online serving goodput实验。

可审计 summary由 clean `72f85ba` 的 `scripts/summarize_p7_graph.py` 重算，输入合同
会拒绝 category缺失、dirty/mixed commit、bucket映射错误、repeat不稳定或 padding
输出污染。focused regression为 `43 passed in 3.90s`。

Raw evidence：

```text
data/p7_graph/p74b_single_image_graph_0fdd4a6.{nsys-rep,sqlite}
data/p7_graph/p74b_single_image_graph_analysis_0fdd4a6.json
data/p7_graph/p74b_single_image_graph_semantic_0fdd4a6.jsonl
data/p7_graph/p74b_padding_fixed8_matrix_00b1012.jsonl
data/p7_graph/p74b_summary_72f85ba.{json,md}
```

### 6.12 P7.5 Projection Fusion：correctness-first 到 full-engine 闭环

P7.4-B trace将 replay中的 linear/GEMV定位为 `253 kernels/step`、
`9.123 ms/step`、`70.55%`。源码映射为每层 attention `q/k/v/o` 四次投影和
MLP `gate/up/down` 三次投影：36 层分别为 `144` 与 `108` 次，另有 1 次
Graph内 linear。P7.5先比较不改变模型数学结构的 projection packing候选。

#### 6.12.1 QKV packing：correctness-first rejection

clean `01b3625` correctness-only probe在同一 BF16输入/权重上比较三次独立投影与
一次 `[Q;K;V]` packed projection。batch1本轮 exact，但 batch `2/4/8` 的 K/V均
不 exact，最大绝对差为 `1.0`，平均绝对差约 `0.079-0.089`；Q在四个 batch中
保持 exact。大 packed矩阵改变了小 K/V GEMV的 cuBLAS算法/舍入，因此该候选在
计时前即被严格门禁拒绝，不报告受污染 GPU上的 speedup。

#### 6.12.2 Gate/up packing：实现与组件 correctness

commit `8767b7a` 将同一输入的 MLP gate/up权重放入一段连续 storage，以一次
`gate_up_proj`完成投影后做 view split。`gate_proj.weight` 与 `up_proj.weight`
仍是该 storage的 Parameter view，因此：

- HF-compatible state-dict仍只有 `gate_proj/up_proj/down_proj`，strict load通过；
- 没有复制 36 层权重或增加实际模型 storage；`Module.to/_apply` 后会重建 view；
- supported forward每层从 `gate + up + down` 三次 linear降为
  `gate_up + down` 两次。

clean `01b3625` 的真实 Qwen shape BF16 correctness覆盖 decode batch
`1/2/4/8`和代表性 prefill rows `210/408/988`，七个 case的 packed/legacy MLP
output均 bitwise exact，max/mean diff均为 `0`。focused model/loader/attention/runner
回归为 `32 passed in 62.19s`。

clean `8293851` 新增显式 `mlp_projection_mode=legacy|packed`，两种执行模式使用同一
packed storage、同一 state-dict与同一 commit，只切换 forward中的两次/一次 gate-up
projection。system benchmark schema-v6记录该字段；clean `021d4e2` 又将它写入
online schema-v2，旧 schema继续可读。

#### 6.12.3 Formal microbenchmark

GPU恢复后，clean `396702d` 在启动 baseline `4 MiB / 0%`、同进程交替A/B、
`warmup=20/repeat=100`下完成七个真实 Qwen shape。所有完整 MLP output bitwise exact：

| Rows | Packed/legacy eager | Packed/legacy Graph |
|---:|---:|---:|
| 1 | `0.9991x` | `0.9859x` |
| 2 | `0.9997x` | `0.9901x` |
| 4 | `0.9971x` | `0.9876x` |
| 8 | `1.0010x` | `0.9918x` |
| 210 | `0.8347x` | `0.8248x` |
| 408 | `0.9978x` | `0.9945x` |
| 988 | `0.9062x` | `0.9041x` |

这是隔离 MLP证据；full-engine结论只使用下一节无 profiler system benchmark。

#### 6.12.4 Full-engine offline 与 E2E correctness

所有记录使用 Qwen3-VL-8B-Instruct config SHA256
`5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661`、RTX 5090
UUID `GPU-989db6f6-3273-d1dd-b2b9-56cced4f30a4`、BF16、off Graph、output32、
`warmup=2/repeat=5`（真实COCO为 correctness-oriented `1/1`）。除 projection mode外
配置不变；8/8 cells的 greedy token hash均 exact。

| Workload | Requests | Legacy TPOT | Packed TPOT | Packed/legacy |
|---|---:|---:|---:|---:|
| text short | 1 | `13.769 ms` | `13.680 ms` | `0.9935x` |
| single image | 1 | `14.054 ms` | `13.962 ms` | `0.9934x` |
| two images | 1 | `14.338 ms` | `14.245 ms` | `0.9935x` |
| video | 1 | `14.383 ms` | `14.304 ms` | `0.9946x` |
| mixed text/image/video | 3 | `15.332 ms` | `15.258 ms` | `0.9952x` |
| real COCO image | 1 | `14.174 ms` | `14.066 ms` | `0.9924x` |
| COCO fidelity A | 4 | `15.531 ms` | `15.450 ms` | `0.9948x` |
| COCO fidelity B | 3 | `15.136 ms` | `15.048 ms` | `0.9942x` |

因此记录 workload上的 unprofiled decode TPOT改善为 `0.483%–0.762%`。single-image
另做 legacy→packed与packed→legacy两轮独立进程，TPOT分别为
`14.052/14.056 ms`与`13.961/13.962 ms`，排除了单次顺序偶然。E2E包含视觉
preprocessing/prefill，部分cell方向不一致；已知TTFT双峰仍在，所以不声明稳定E2E加速。

#### 6.12.5 HF、online 与完整回归

- single-image、multi-image、video各32-token teacher-forced model-precision logits
  相对HF的max/mean diff均为`0`，PPL diff均为`0`；750/750权重加载且无missing/unexpected；
- single-image rate4与mixed-rate10 online A/B逐请求token exact，双方SLO goodput
  fraction均为`1.0`；mixed两边peak active均为`5`。每个online cell只有一次process run，
  不据此声称goodput speedup；
- fresh editable venv成功运行`example.py`并输出8 tokens；
- clean `021d4e2`完整JUnit为`287 tests / 0 failures / 0 errors / 6 skipped`，即
  `281 passed, 6 skipped in 297.622s`。

#### 6.12.6 Node-level Systems confirmation

clean `021d4e2` 对legacy/packed分别采集31个measured Graph replay，CUDA Graph
node tracing只覆盖`cudaProfilerStart/Stop`区间：

| Metric / replay | Legacy | Packed | Delta |
|---|---:|---:|---:|
| all kernels | `2,000` | `1,964` | `-36` |
| linear/GEMV kernels | `253` | `217` | `-36` |
| kernel busy | `12.815 ms` | `12.721 ms` | `-0.093 ms` |
| linear/GEMV time | `9.087 ms` | `8.999 ms` | `-0.088 ms` |

理论映射得到真实trace逐项确认：36层各减少一个gate/up linear，没有额外kernel回归。
最终判定为保留packed默认，但把claim限制在上述小幅decode TPOT收益。

Raw evidence：

```text
data/p7_optimization/p75_qkv_correctness_01b3625.json
data/p7_optimization/p75_packed_mlp_shape_correctness_01b3625.json
data/p7_optimization/p75_packed_mlp_micro_396702d.json
data/p7_optimization/p75_*_off_graph_{legacy,packed}_{8293851,021d4e2}.jsonl
data/p7_optimization/p75_online_*_{legacy,packed}_021d4e2.json
data/p7_optimization/p75_single_image_graph_{legacy,packed}_021d4e2.{nsys-rep,sqlite}
data/p7_optimization/p75_single_image_graph_{legacy,packed}_analysis_021d4e2.json
data/p7_optimization/p75_hf_logits_ppl_8293851.{xml,stdout.txt}
data/p7_optimization/p75_summary_021d4e2.{json,md}
data/p8_delivery/example_fresh_021d4e2.stdout.txt
data/p8_delivery/final_full_regression_021d4e2.xml
```
