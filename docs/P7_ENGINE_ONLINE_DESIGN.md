# P7.2/P7.3 Engine Contracts 与 Online Serving 设计

> 日期: 2026-07-16
>
> P7.2 状态: `COMPLETE`（commit `8b27edc`）
>
> P7.3 状态: `IMPLEMENTING`

## 1. 目标与边界

P7.2 先把原先集中在 `LLMEngine.step()` 和 `Scheduler.schedule()` 的可变控制流
拆成可测试合同；P7.3 在这些合同上增加 arrival、continuous batching、admission、
preemption 与 SLO metrics。模型 forward、KV 物理布局和已有 public generate API
不因架构重构改变。

本阶段不把 offline replicated batch 当作 online serving，也不把请求提交数当作
并发数。online 记录必须保留每个请求的提交、首次调度、首 token、逐 token 和完成
时间。

## 2. P7.2 已冻结合同

| 边界 | 实现 | 不变量 |
|---|---|---|
| Request FSM | `engine/request.py` | 只有显式合法 transition；terminal request 不可复活 |
| Immutable batch handoff | `engine/contracts.py::BatchPlan` | phase、成员、token budget、KV transfer 创建后不可修改 |
| Scheduler policy | `engine/scheduler_policy.py` | admission/chunk/preemption 决策与队列、CUDA 状态分离 |
| KV manager | `KVCacheManager` Protocol + `BlockManager` | scheduler 只消费 ownership/capacity contract |
| Executor | `engine/executor.py::ModelExecutor` | 按 plan 执行 CoW/swap/model/compaction，不自行改调度 |
| Metrics | `MetricsSink` + `EngineMetrics` | 只观察事件，不反向驱动 scheduler |

`Scheduler.schedule()` 现在返回 `BatchPlan`。P1-P6 的五元组解包仍通过
`BatchPlan.__iter__()` 兼容；主引擎只使用 named fields。`LLMEngine.step_result()`
返回强类型 `StepResult`，原 `step()` 返回保持不变。

P7.2 同时修复两条资源生命周期问题：

- 取消 swapped request 时归还 CPU KV pages。
- `LLMEngine.exit()` 先释放 executor 对 runner 的引用，避免同进程下一次模型加载
  残留整套权重/KV cache。

## 3. P7.3 Online request contract

online request 至少包含：

- 稳定 request id、arrival offset、请求类型和 sampling params。
- admission 结果；拒绝必须给出有限、可聚合的原因。
- terminal 状态：finished、cancelled 或 rejected。
- prompt/output token 数及 request-level queue/TTFT/TPOT/latency。

Engine loop 在下一 arrival 前只执行当前可运行 batch；arrival 到期后立即提交，后续
batch 可增减 request，形成 continuous batching。无请求时允许等待下一 arrival，
不能调用空 scheduler step。

## 4. 调度要求

1. FCFS 是首个稳定 policy，但 policy 选择不能散落在 engine/executor。
2. 长 prefill 使用 immutable per-request token budget；ModelRunner 不得重新猜 chunk。
3. text/image/multi-image/video 可以进入同一 admission/queue/metrics 路径。
4. decode 不得因持续 arrival 无限饥饿；prefill/decode interleave policy 必须可记录。
5. KV 不足时保留 swap/recompute preemption，并分别计数。
6. 超长、队列满和不支持组合必须 fail closed。

## 5. Online benchmark schema

每条正式 record 必须包含：

- arrival process、request rate、seed、duration/request count。
- model/GPU/commit/clean state、KV pool、block size、execution/compression config。
- request-level queue、TTFT、TPOT、latency 与 finish reason。
- p50/p90/p99 queue、TTFT、TPOT、latency。
- request/output throughput、goodput 及明确 SLO thresholds。
- peak waiting/running/swapped、KV occupancy、swap/recompute preemption、rejection。

goodput 只统计同时满足 TTFT 与 TPOT SLO 的完成请求；不能由平均 throughput 推导。

## 6. P7.2 验证

clean `8b27edc`：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
.venv-local/bin/python -m pytest -q \
  --junitxml=data/p7_engine/p72_full_regression_8b27edc.xml
```

JUnit：`tests=255`、`failures=0`、`errors=0`、`skipped=6`、
`time=239.200s`，即 `249 passed, 6 skipped`。

## 7. P7.3 出口

- scheduler/arrival/metrics 的 deterministic unit tests。
- text 与 VL online integration，输出与等价 offline reference满足既有数值合同。
- queue/admission/cancel/swap/recompute/chunked paths均有显式测试。
- clean 单卡 online matrix 与结构化 summary。
- full regression 通过，README/ROADMAP/VERIFICATION/claim边界同步。
