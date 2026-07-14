# Prism-Infer 性能报告

> 更新日期: 2026-07-11
> 当前阶段: P6.1 统一 benchmark contract
> 报告性质: runner validation baseline，不是正式对外性能结论

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

## 4. 当前限制与下一门禁

- 当前记录为 dirty-worktree runner validation；提交后需要同命令 clean rerun。
- 只测了 synthetic single-image；manifest 其余 text/multi-image/video/mixed cases 尚未形成 P6 baseline 表。
- 只测 8-token 输出；长 decode、长 visual context、batch/concurrency matrix 和 OOM boundary 尚未测量。
- 没有真实任务 quality metric、teacher-forced logits/ppl 或稳定前缀矩阵。
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
