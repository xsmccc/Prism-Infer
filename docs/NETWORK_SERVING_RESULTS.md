# Prism-Infer 原生网络服务与视觉调度结果

> 更新日期：2026-07-29
> 证据基线：`25eeb72d0da8dc462445e9dbf46a14a8907a7bc9` + 本文对应 dirty
> candidate；最终实现以本文所在提交为准
> GPU：NVIDIA GeForce RTX 5090，UUID
> `GPU-298199be-71f2-7400-1d23-5f71e3c3d743`
> Driver/CUDA/PyTorch：580.105.08 / 13.0 / 2.11.0+cu130
> Model：Qwen3-VL-8B-Instruct
> `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`

## 1. 结论

这轮工作补齐了真实 HTTP/SSE 服务边界，并用相同 arrival trace 定位在线瓶颈。最终
结论不是“Prism 在线超过 vLLM/SGLang”：

- 原生服务支持 JSON 与 SSE token streaming、bounded ingress、断连取消、请求级
  资源释放和单 engine owner；60-request 对照中，网络与进程内 raw throughput
  仅差 `+0.049%`，HTTP/SSE 不是当前主要瓶颈。
- loaded profile 的主要阻塞是原子视觉 prefill：文本约 `50 ms`，单图稳定后约
  `65 ms`，H1 八图约 `204–212 ms`，H2 视频约 `232 ms`；它会插入 decode
  cadence，而不是网络层拖慢 token。
- 动态视觉 Tensor CUDA Graph 在这张 RTX 5090 的 mixed-shape loaded trace 中产生
  可复现错误输出。关闭它后，600-request 轨迹中异常首 token `2→0`、显存峰值
  `24,456→24,018 MiB`、raw throughput `+0.88%`、TPOT p50 `-1.36%`，但
  TTFT/E2E p50 和 SLO goodput 退化。因此这是正确性与稳定性修复，不是全面性能优化。
- 视觉感知调度能改善中位延迟，但未改善最终 SLO goodput。加入严格旁路上限后，
  TTFT p50/p90 相对 FCFS 改善 `14.5%/3.9%`，E2E p50/p90 改善
  `3.3%/1.6%`，raw throughput 只下降 `0.47%`；但 goodput 仍下降 `15.8%`。
  它保留为显式实验策略，不作为默认配置。
- 同卡 60-request 开发对照中，Prism loaded goodput 仍明显低于 vLLM/SGLang。
  Prism 当前可使用的外部领先结论仍只限冻结 H1/H2 batch1 offline TPOT。

默认在线配置因此选择：

```text
FCFS
+ decode torch.compile / CUDA Graph
+ visual compaction / scaled-FP8 KV
- dynamic vision tensor CUDA Graph
```

## 2. 服务边界

### 2.1 已实现

- `POST /v1/generate`：非流式 JSON 与 SSE 流式 token；
- `GET /health`：engine owner 存活检查；
- 单独线程拥有 engine，网络协程不直接并发修改 scheduler；
- bounded ingress 与明确 overload 响应；
- 客户端断开后取消 waiting/running request，并释放 KV；
- benchmark 从 HTTP 客户端记录 TTFT、TPOT、E2E、token IDs、arrival trace、
  class-aware SLO goodput 和进程级 NVML peak；
- 视频输入使用明确的已采样 frame payload，不把服务端隐式抽帧差异混入结果。

### 2.2 当前边界

- 这是研究用原生 API，不是 OpenAI-compatible API；
- 没有多进程 frontend、认证、限流集群、PD 分离或跨机容错；
- 外部 vLLM/SGLang 开发对照仍使用各自原生进程内入口，不能写成完全同协议的
  network-serving 排名；
- 60-request 是候选选择；本文的最终取舍使用 600-request frozen H3 单种子结果，
  不是跨种子置信区间。

## 3. Workload 与指标

600-request 最终轨迹使用：

- conditional-video H3：text 40%、single-image 30%、H1 eight-image 20%、
  H2 16-frame video 10%；
- Poisson arrival，rate 4 req/s，seed `20260717`；
- greedy，output 64，max model length 4096；
- class SLO 来自冻结的 vLLM rate-1 p50：
  `TTFT SLO = 5×p50`，`TPOT SLO = 2×p50`；
- Goodput 只统计同时满足本类 TTFT 和 TPOT SLO 的输出 token；
- graph on/off、FCFS/vision-aware 的 arrival trace SHA256 均为
  `105fa73b203c42dc61b96be60d367b7b567cc78ccd4183ca9193280fffdf4235`，
  且 `full_frozen_h3=true`。

Goodput 对临界值很敏感。中位数更好不保证 goodput 更高：只要 TPOT 略过阈值，或
重视觉请求的尾部等待变长，该请求的全部 output token 都不会计入 goodput。

## 4. Profiling 归因

10-request semantic profile 记录 142 个 engine steps：

| Step 类型 | 典型时长 | 解释 |
|---|---:|---|
| decode | `11–15 ms` | 高频、直接决定 token cadence |
| text prefill | `~50 ms` | 轻请求 |
| single-image prefill | `~65 ms` | shape warmup 后 |
| H1 eight-image prefill | `204–212 ms` | 原子重视觉批次 |
| H2 video prefill | `~232 ms` | 最长原子视觉批次 |

旧 FCFS 可以连续出现“重视觉 prefill → 一个 decode → 下一个重视觉 prefill”。这解释
了 loaded TPOT 和排队 TTFT 同时恶化，也解释了为什么仅优化 HTTP、序列化或单请求
H1 TTFT不能解决 goodput。

## 5. 网络边界与外部开发对照

同一 60-request trace：

| 系统 | Raw tok/s | SLO goodput tok/s | TTFT p50 | TPOT p50 | E2E p50 | NVML peak |
|---|---:|---:|---:|---:|---:|---:|
| Prism in-process | 214.398 | 28.586 | 363.814 ms | 27.755 ms | 2,151.135 ms | 24,456 MiB |
| Prism HTTP/SSE | 214.503 | 32.176 | 330.792 ms | 27.601 ms | 2,089.673 ms | 24,456 MiB |
| vLLM development | 222.462 | 211.339 | 145.463 ms | 13.659 ms | 1,018.355 ms | 23,826 MiB |
| SGLang development | 221.181 | 191.690 | 161.562 ms | 14.500 ms | 1,118.874 ms | 23,666 MiB |

Prism network 相对 in-process raw throughput 为 `+0.049%`，属于运行波动。外部两行
与 Prism 的请求类型和 arrival offsets 完全一致，但 frontend protocol 不同，只能
用于同卡开发定位。它们明确说明当前差距在 engine token cadence，而不在 HTTP。

## 6. 动态视觉 Graph：拒绝作为默认路径

FCFS、同一 600-request network trace：

| 配置 | Raw tok/s | Goodput tok/s | Good req | TTFT p50/p90 | TPOT p50/p90 | E2E p50/p90 | Peak | 异常首 token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visual Graph on | 230.666 | 11.918 | 31/600 | 1,310/4,275 ms | 30.769/37.047 ms | 3,270/6,461 ms | 24,456 MiB | 2 |
| visual Graph off | 232.686 | 10.083 | 26/600 | 1,769/4,028 ms | 30.350/36.864 ms | 3,645/6,006 ms | 24,018 MiB | 0 |

关闭相对开启：

- raw throughput `+0.88%`；
- TPOT p50/p90 `-1.36%/-0.49%`；
- TTFT p90 `-5.78%`，E2E p90 `-7.04%`；
- peak memory `-438 MiB`；
- TTFT p50 `+34.96%`，E2E p50 `+11.48%`；
- goodput `-15.39%`。

两个 graph-on 异常不是正常随机分叉：

- H1 request `formal-00027` 的首 token 为 `0`，graph-off 对应 token 为 `6025`；
- single-image request `formal-00028` 的 64 个输出 token 全为 `0`，graph-off
  对应输出以 `[785, 2168, 374, ...]` 开始。

候选选择期还观察到 measured run 内首次捕获新视觉 shape，并在同一 prefill batch
出现零 token。现有证据足以把动态 mixed-shape graph 与错误隔离，但不足以声称已经
定位到某个具体 CUDA kernel。旧 fixed-shape clean H1 Graph 结果仍是有效的历史受限
结果；不能外推到动态 loaded serving。

## 7. 视觉感知调度：保留实验实现，拒绝默认启用

策略将 `>=4096` raw vision patches 的请求视为 heavy。heavy head 在 decode credit
不足时允许最老的 light request 旁路；最终版本增加
`max_light_prefill_bypasses_per_heavy=2`，确保队首 heavy 最多被越过两次。

同一 600-request network trace、visual Graph off：

| 策略 | Raw tok/s | Goodput tok/s | Good req | TTFT p50/p90 | TPOT p50/p90 | E2E p50/p90 |
|---|---:|---:|---:|---:|---:|---:|
| FCFS | 232.686 | 10.083 | 26/600 | 1,769/4,028 ms | 30.350/36.864 ms | 3,645/6,006 ms |
| vision-aware，无旁路上限 | 217.634 | 5.078 | 14/600 | 694/9,828 ms | 28.889/34.668 ms | 2,612/11,606 ms |
| vision-aware，旁路上限 2 | 231.583 | 8.491 | 22/600 | 1,512/3,871 ms | 30.721/36.706 ms | 3,527/5,909 ms |

无上限候选把 TTFT p50 降低 `60.8%`，但重视觉到达速率超过策略为 heavy 提供的服务
速率，TTFT p90 增加 `144.0%`、goodput 降低 `49.6%`。这是典型 starvation。

旁路上限修复了尾部：

- TTFT p50/p90 `-14.51%/-3.91%`；
- E2E p50/p90 `-3.26%/-1.61%`；
- TPOT p90 `-0.43%`；
- raw throughput `-0.47%`；
- TPOT p50 `+1.22%`，goodput `-15.79%`。

所以最终判断是：它是可解释的 latency-biased trade-off，不是满足当前双 SLO 的
goodput 优化。继续针对这一条 trace 调 threshold/interval/bypass 会过拟合，故停止
调参，默认保持 FCFS。

## 8. 失败尝试与取舍

| 尝试 | 观察 | 决策 |
|---|---|---|
| 把差距归因于 HTTP/SSE | network/in-process raw 仅差 `0.049%` | 否决；转向 engine profile |
| P13 phase-decomposed prefill | 单次最大 prefill 变短，但 goodput `-34.18%`、TTFT/TPOT 退化 | 候选代码删除 |
| vision-aware interval 32，无 starvation cap | p50 很好，p90 与 goodput 崩溃 | 否决 |
| 加入 heavy 旁路上限 2 | 尾部恢复，goodput仍 `-15.79%` | 只保留实验策略 |
| 动态视觉 Tensor Graph | 某些中位数更好，但出现错误 token并多占显存 | 默认关闭 |
| 继续做通用 GEMV kernel | 当前证据指向调度与 loaded decode cadence，且 vendor GEMV 已高度优化 | 暂不投入 |

## 9. 面试讲法

推荐按“现象—假设—实验—反例—取舍”讲：

1. 我先实现真实 HTTP/SSE，而不是继续用进程内数字猜网络开销。
2. 相同 arrival trace 表明网络开销几乎为零；semantic profile 显示
   `200+ ms` 原子视觉 prefill 插入 `11–15 ms` decode。
3. 我做了 heavy/light 调度，短 trace 中 TTFT 和一部分 goodput 看起来改善。
4. 真实 600-request steady-state 暴露 starvation：中位数下降不代表尾部或
   goodput 改善。
5. 我给旁路加严格上限，修复 p90，却仍因为 TPOT 临界分布导致 goodput下降，所以
   不把它设成默认。
6. 独立 graph on/off 又发现动态视觉 Graph 的错误 token。最终保留 decode Graph，
   关闭动态视觉 Graph，用 `438 MiB` 显存下降、零异常输出和完整负面指标共同说明
   这个决定。

常见追问：

- **为什么 Goodput 与 p50 方向不同？** Goodput 是每请求同时通过两类 SLO 的离散
  计数；临界 TPOT 和 heavy tail 比整体中位数更重要。
- **为什么不继续调参？** 当前 workload 比例已参与 cap=2 的设计；继续针对单 trace
  搜索会变成 workload overfitting，不能证明通用调度收益。
- **为什么还保留实验策略？** 它提供了有界重排实现和清晰 latency/goodput
  trade-off，可用于未来明确选择 latency-biased policy 的场景；默认不启用。
- **CUDA Graph 为什么会错？** 已通过唯一变量 graph on/off 和相同 request 定位到
  动态视觉 Graph 路径；可能与 lazy shape capture/地址生命周期有关，但没有足够
  kernel 证据时不声称更具体的根因。
- **当前真正的下一瓶颈？** 外部开发对照的 TPOT 约 `13.7–14.5 ms`，Prism loaded
  TPOT 约 `27–31 ms`。下一阶段应 profile batch `2/4/8` 的 decode Graph 实际
  residency、launch gap 与 GEMV/attention占比，而不是继续重排 prefill。

## 10. Evidence

服务器原始记录：

- `data/network_serving/profile_online_h3_period10.json`
- `data/network_serving/dev_inprocess_h3_conditional_r4_s20260717_n60.json`
- `data/network_serving/dev_network_h3_conditional_r4_s20260717_n60.json`
- `data/network_serving/dev_vllm_h3_conditional_r4_s20260717_n60.json`
- `data/network_serving/dev_sglang_h3_conditional_r4_s20260717_n60.json`
- `data/network_serving/dev_network_fcfs_vision_graph_h3_conditional_r4_s20260717_n600.json`
- `data/network_serving/dev_network_fcfs_no_vision_graph_h3_conditional_r4_s20260717_n600.json`
- `data/network_serving/dev_network_vision_aware_no_vision_graph_h3_conditional_r4_s20260717_n600.json`
- `data/network_serving/dev_network_vision_aware_capped_no_vision_graph_h3_conditional_r4_s20260717_n600.json`

所有 server/benchmark 进程均已正常停止，验证后 GPU process memory 为 `0 MiB`。
