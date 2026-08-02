# 重复视觉上下文：设计、实验与取舍

## 1. 解决什么问题

目标负载是“上传一组图片后连续提问”。第一次请求需要运行 Processor、Vision Encoder、
DeepStack 和语言 Prefill；后续问题的媒体不变，但问题文本不同。

只缓存视觉 embedding 只能省去 Vision/DeepStack，语言模型仍要为每个问题重算长视觉
前缀。保存 Dense Prefix KV 可以跳过这部分 Prefill，但视觉 token 很快占满有限的 KV
Pool。Prism-Infer 把两件事接在一起：

1. 用 Scaled-FP8 存储公共 Prefix KV；
2. 物理删除一部分视觉 KV，让单个 Prefix 占用更少 pages；
3. 后续问题直接挂接只读共享页，只计算问题后缀；
4. 活跃请求需要空间时，从同一个 KV Pool 回收缓存页。

这条路径的价值不在于让一次 Attention 更快，而在于固定 KV 预算下驻留更多媒体，减少
Prefix 淘汰和冷重算。视觉 token 删除是有损操作，因此性能和质量必须同时报告。

## 2. 请求路径

### 2.1 可复用 Prompt

图片放在问题之前，并保留明确编号：

```text
Image 1: <image>
Image 2: <image>
...
Question: ...
```

最后一个视觉占位符之前的精确 token 序列是公共 Prefix；问题及生成 token 不进入缓存。
编号不能省略，因为多图问题经常引用图片顺序。最终布局在 85 条 MuirBench 多问题样本上
得到 46/85，官方交错布局为 49/85，仍有 3 题差距。

### 2.2 Prefix-first

```text
media identity + processor layout + common-prefix tokens
  -> direct Prefix Cache lookup
  -> hit: attach shared pages and skip Vision/DeepStack hydration
  -> miss: hydrate/build Vision/DeepStack, prefill, compact and retain pages
  -> Scheduler admission
```

查询发生在 Scheduler admission 之前。旧路径先调用视觉缓存恢复，再判断 Prefix 是否
命中；即使命中也已经付出 hydration 成本。新路径在命中时不调用
`hydrate_visual_embedding_cache()`，并记录：

- `pre_admission_hits`：admission 前发现可复用 Prefix；
- `visual_hydration_skips`：因此跳过的视觉恢复次数；
- `stale_probe_fallbacks`：早期命中后、实际分配前条目被淘汰的次数。

原始媒体 Tensor 保留到 Prefix pages 真正分配完成。如果早期命中的条目在等待期间被
回收，请求使用原始媒体走完整冷路径，不会拿到悬空页。

### 2.3 O(1) 内容寻址

Prefix ID 由以下内容直接构造：

```text
model / processor layout
+ ordered media SHA256
+ common-prefix token count
+ full common-prefix token SHA256
```

字典查找代替遍历全部 Prefix Entries。命中后仍比较媒体 key、公共长度和完整 token
序列；摘要碰撞会报错，不会复用错误 KV。媒体身份来自内容，不依赖文件路径或 Python
对象地址。

### 2.4 页所有权与全池回收

旧实现最多允许 Prefix Cache 占 `min(256, num_blocks // 8)` pages；在 220-page 实验
中只剩 27 pages，无法形成有意义的视觉工作集。现在 Prefix Cache 可以借用全部暂时
空闲的 KV pages。

缓存中的完整页只读共享。未满尾页不能直接共享写入，请求获得自己的 tail clone；追加
token 时按需 Copy-on-Write。活跃请求的分配、CoW 或 Swap-in 缺页时，先回收空闲 tail
clone，再按下式淘汰完整 Prefix Entry：

```text
benefit_tokens * (1 + lifetime_hits) / resident_pages
```

引用计数不为零的共享页不会释放。本轮没有增加新的淘汰算法或做参数扫描；重点是正确
打通全池借用、页所有权和回收路径。

## 3. 为什么主路径使用 Uniform

运行时支持 Uniform 与 Attention Top-k。Attention Top-k 依赖当前问题：每道问题重新
选择可以使用问题信息，但必须重新 Prefill，无法复用 Prefix KV；沿用第一道问题的选择
可以复用，却可能删除后续问题需要的区域。

主路径采用 query-agnostic Uniform，使同一份压实页对所有后续问题有效。图片配置为
`keep_ratio=0.6`、最少保留 768 个视觉 token；256-token page 粒度决定了实际释放量。
视频 token 删除默认关闭，只在质量对照中显式设置 256-token 最小保留值。

Uniform 不是新的 token 选择算法。项目贡献是把问题无关选择、Scaled-FP8 Paged KV、
物理压实、共享页所有权和跨问题复用接成一条完整运行时路径。

## 4. 实验协议

| 项目 | 配置 |
|---|---|
| Model | Qwen3-VL-8B-Instruct，固定 revision |
| Device | RTX 5090，TP1 |
| Page / context | 256 tokens/page，`max_model_len=8192` |
| KV budget | 220 Prism Scaled-FP8 pages / 4,282,122,240 bytes |
| Traffic | 600 requests，Zipf-1.0，Poisson 4 req/s，greedy，output 16 |
| Prism | Scaled-FP8，256 MiB Vision Cache，现有 compile/CUDA Graph 路径 |
| vLLM | 0.25.1，FP8 KV、APC、Processor/Encoder Cache |
| SGLang | 0.5.15.post1，FP8 KV、Radix Cache、`mm_global_cache` |

MuirBench 按完全相同的有序媒体 SHA256 分组，再按内容哈希排序，不根据性能结果挑样本。
Dense Scaled-FP8 预运行给出每组真实 Prefix pages，据此构造：

| Workset | Media groups | Dense Prefix pages | 相对 220-page 预算 |
|---|---:|---:|---:|
| `fit` | 25 | 151 | 68.6% |
| `knee` | 38 | 223 | 101.4% |
| `pressure` | 59 | 333 | 151.4% |

每组先请求一次建立工作集，随后三套引擎消费相同图片、prompt token、请求顺序、到达
时间和生成参数。外部引擎没有与 Prism 等价的“当前驻留媒体条目数”计数，因此该字段不
从命中率反推。三者输出吞吐都约为 64.1–64.2 tok/s，因为固定到达率与 output 16 主导
了该指标，不能据此声称吞吐领先。

## 5. 性能结果

### 5.1 Prism 内部因果链

`pressure` 工作集：

| 路径 | 驻留媒体 | Prefix 淘汰 | Vision miss | 重算 prompt tokens | TTFT p50 / p99 | E2E p50 / p99 |
|---|---:|---:|---:|---:|---:|---:|
| Vision/DeepStack Cache only | 7 | 0 | 394 | 1,052,583 | 578.965 / 3,044.275 ms | 1,832.086 / 4,646.051 ms |
| Dense Prefix | 33 | 110 | 91 | 192,090 | 124.009 / 707.520 ms | 420.122 / 1,179.932 ms |
| Compact Prefix | **48** | **33** | **17** | **83,018** | **102.401 / 270.588 ms** | **350.512 / 774.318 ms** |

Compact 运行本身产生 4,412 个 dense-equivalent Prefix pages，物理保留 2,910 个，减少
34.0%。相对 Dense Prefix 路径，驻留媒体增加 45.5%，淘汰减少 70.0%，Vision miss
减少 81.3%，重算 prompt tokens 减少 56.8%；TTFT p50/p99 分别降低 17.4%/61.8%。

这里的 page 数是整个运行期间累计 admission 的页数，不是结束时仍驻留的页数，因此
Compact 的 4,412→2,910 应在同一次运行内部比较，不能直接拿 2,910 与 Dense 路径的
3,907 做压缩率。

### 5.2 与外部引擎比较

| Workset | Engine | TTFT p50 / p99 | E2E p50 / p99 | Process peak | 重算 prompt tokens |
|---|---|---:|---:|---:|---:|
| fit | Prism Compact | **97.693** / 253.682 ms | 330.853 / 690.553 ms | 23,998 MiB | 64,222 |
| fit | vLLM | 112.978 / **220.883 ms** | **291.372 / 449.895 ms** | **23,628 MiB** | **50,020** |
| fit | SGLang | 428.032 / 1,176.561 ms | 739.369 / 1,911.111 ms | 23,944 MiB | 50,020 |
| knee | Prism Compact | **97.837** / 218.424 ms | 333.900 / 701.338 ms | 24,002 MiB | 63,392 |
| knee | vLLM | 109.921 / **200.345 ms** | **286.894 / 419.819 ms** | **23,628 MiB** | **50,547** |
| knee | SGLang | 430.673 / 1,614.508 ms | 704.202 / 2,172.660 ms | 23,944 MiB | 50,547 |
| pressure | Prism Compact | **102.401 / 270.588 ms** | 350.512 / **774.318 ms** | 24,006 MiB | **83,018** |
| pressure | vLLM | 131.483 / 581.976 ms | **323.228** / 886.035 ms | **23,714 MiB** | 165,031 |
| pressure | SGLang | 494.911 / 1,623.782 ms | 855.944 / 2,270.308 ms | 24,284 MiB | 179,111 |

在 `pressure` 上，Prism 相对 vLLM 的 TTFT p50/p99 低 22.1%/53.5%，E2E p99 低
12.6%，重算 prompt tokens 少 49.7%；E2E p50 慢 8.4%，进程峰值多 292 MiB。
`fit` 和 `knee` 上 vLLM 的 E2E 与部分 tail latency 更好，因此结果不是“所有工作集、
所有指标都领先”。更重要的是 Compact 是有损配置，下一节的质量差异必须与这张表一起
解释；它不是等质量的引擎排名。

## 6. 质量结果

### 6.1 Prompt 布局

| Dense layout | MuirBench official-compatible accuracy |
|---|---:|
| 官方交错图片与文本 | 49/85（57.65%） |
| 带图片编号的 media-first | 46/85（54.12%） |

这 3 题差距来自 Prompt 重排，不是 KV 压实。后续压实质量都以 Dense labeled
media-first 为配对参考，避免混淆两个变量。

### 6.2 实际删除 token 的 MuirBench 样本

| 选择方式 | 能否跨问题复用 | 正确数 | 相对 Dense |
|---|---:|---:|---:|
| Dense labeled media-first | 是 | **27/49（55.10%）** | — |
| 每题 Attention Top-k | 否 | 20/49（40.82%） | -7 |
| 第一题 Attention Top-k 沿用 | 是 | 20/49（40.82%） | -7 |
| Query-agnostic Uniform | 是 | 20/49（40.82%） | -7 |

四组使用完全相同的 49 条样本；三个压实方案都实际删除 52,120 个视觉 token。Uniform
相对 Dense 有 15 条从对变错、8 条从错变对，净减少 7 个正确答案。每题 Attention 在
这一预算下没有恢复 aggregate accuracy，同时失去跨问题复用；沿用第一题选择也没有质量
优势。因此当前结果支持 Uniform 的复用语义，却不支持“0.6 保留率基本无损”的说法。

### 6.3 DocVQA 与 MVBench

| Dataset | Dense | Uniform | 实际删除样本 | 结论 |
|---|---:|---:|---:|---|
| DocVQA，190 条 | ANLS 0.93335 | ANLS 0.93335 | 0 | 768-token 图片下限阻止了删除，不能证明压实无损 |
| MVBench，252 条 | 183/252（72.62%） | 113/252（44.84%） | 252 | 视频压实明显损伤质量，默认关闭 |

MVBench 来自 123 个精确同视频多问题组，媒体文件逐个记录 archive revision、CRC 和
SHA256。Uniform 对这些样本共删除 20,064 个视觉 token。该结果说明图片上的配置不能
直接迁移到视频；项目不把视频纳入性能排名，也不默认启用视频删除。

## 7. Trace 证据

Nsight Systems capture 包含一次 cold request 和同媒体、不同问题的 Prefix hit。它用于
核对执行路径，不用作主性能表：

| 检查 | 结果 |
|---|---|
| Cold range 中出现 Vision | `prism::model.vision.embedding_cache_miss` × 1 |
| Prefix-hit range 中出现 Vision/DeepStack | 0 |
| Prefix-hit `visual_hydration_skips` | +1 |
| Prefix-hit `stale_probe_fallbacks` | 0 |
| 公共 Prefix 复用 | 145 / 275 prompt tokens（52.73%） |
| Trace 审计 | 6/6 checks passed |

Profiler 下 cold/hit 的 CPU range 分别为 460.587/266.810 ms，GPU busy time 分别为
44.365/19.576 ms。Prefix hit 仍需计算问题后缀和生成 15 个 token；证据只说明公共
Prefix 与视觉路径被跳过，不表示整个请求没有 Prefill。原始 `.nsys-rep`、JSON 摘要和
审计结果位于 `artifacts/working_set/trace/`。

## 8. 实现过程中解决的问题

1. **Prefix 查询太晚。** 命中前已经进行视觉 hydration。查询移到 admission 前后，命中
   请求才真正绕过 Vision/DeepStack。
2. **Prefix 驻留上限太小。** 八分之一 KV Pool 的固定限制让缓存容量与真实空闲空间无关。
   改为全池借用，并在活跃请求缺页时回收。
3. **查找随条目数线性增长。** 精确身份已经存在，因此改为摘要字典直查，并保留完整 token
   校验。
4. **早期命中可能过期。** 查询与页分配之间条目可能被淘汰。保留原始媒体并实现 stale
   fallback，使请求安全回到冷路径。
5. **共享尾页会破坏所有权。** 完整页只读共享，未满尾页使用 clone 和 CoW；Swap、追加、
   物理压实与淘汰都遵守同一引用计数。
6. **Attention 选择依赖问题。** 每题选择无法复用，第一题选择复用也没有质量优势；主路径
   使用问题无关 Uniform，并把质量损失单独量化。
7. **质量记录曾读取错误阶段。** 问题后缀继续生成时会覆盖临时 decision record。最终从
   Prefix Entry 读取真正驻留的 compression record，只统计物理删除过 token 的样本。
8. **长矩阵遇到实例级中断。** 续跑逻辑逐项校验已完成 cell 的运行身份和 SHA256，只运行
   未完成 cell，并保留中断日志；最终 15/15 性能 cell 与 9/9 质量 stage 完整。

## 9. 代码路径

| 内容 | 主要文件 |
|---|---|
| Prefix-first submit、hydration skip、在线记录 | `prism_infer/engine/online.py`、`prism_infer/engine/llm_engine.py`、`prism_infer/engine/metrics.py` |
| Prefix identity、页引用、全池回收、tail CoW | `prism_infer/engine/block_manager.py`、`prism_infer/engine/sequence.py`、`prism_infer/engine/contracts.py` |
| 视觉选择与物理 KV 压实 | `prism_infer/engine/visual_pruning.py`、`prism_infer/engine/compression.py`、`prism_infer/engine/kv_compaction_coordinator.py`、`prism_infer/engine/model_runner.py` |
| 共用 plan 与三引擎矩阵 | `benchmarks/build_working_set_plan.py`、`benchmarks/working_set_workload.py`、`benchmarks/run_working_set_matrix.py` |
| 质量配对与结果汇总 | `benchmarks/bench_working_set_quality.py`、`prism_infer/analysis/working_set_quality.py`、`benchmarks/summarize_working_set.py` |
| Trace 采集与审计 | `benchmarks/trace_working_set_prefix.py`、`benchmarks/analyze_nsys.py`、`benchmarks/audit_working_set_prefix_trace.py` |

## 10. 面试时怎么讲

建议按“场景—瓶颈—实现—证据—取舍”展开，而不是先罗列组件：

1. **场景。** 多模态聊天中，同一组图片会被连续提问。Vision Cache 只能省 Vision，
   Dense Prefix Cache 又会被长视觉 KV 很快占满。
2. **瓶颈。** 固定 KV 预算下，真正影响 TTFT 的不是一次 Attention Kernel，而是媒体条目
   驻留不足后反复发生的 Vision 与公共 Prefix Prefill。
3. **实现。** 把有序媒体移到问题之前，使用内容寻址在 admission 前查 Prefix；缓存保存
   Scaled-FP8、物理压实的只读 pages，并处理 tail CoW、引用计数、Swap 与缺页回收。
4. **证据。** 在 `pressure` 工作集上，pages 4,412→2,910，驻留媒体 33→48，重算
   tokens 192,090→83,018，TTFT p50/p99 124.009/707.520→102.401/270.588 ms；Trace
   证明命中路径没有 Vision/DeepStack。
5. **取舍。** `keep_ratio=0.6` 在实际删 token 的 MuirBench 样本上从 27/49 降到 20/49，
   所以它是容量—延迟—质量 operating point，不是无损加速；视频下降更明显，因此默认
   不删视频 token。

如果被问“与 vLLM/SGLang 的区别”，应回答：通用框架已有 Processor/Encoder Cache 和
文本 Prefix Cache；本项目的特色是把问题无关视觉压实、Scaled-FP8 Paged KV 和跨问题
Prefix 页复用接在同一页生命周期里，并用相同请求 plan 验证工作集压力下的收益。不要说
“全面超过”；准确说法是固定场景下 TTFT/tail E2E 更好，但 E2E p50、显存和质量各有代价。

## 11. 可以陈述的结论与限制

- 已验证的系统因果链是：pages 减少 → 驻留媒体增加 → 淘汰与重算减少 → `pressure`
  工作集的 TTFT 和 tail E2E 改善。
- 结果覆盖 Qwen3-VL-8B、RTX 5090、TP1、固定 4 GiB KV 预算和带编号 media-first
  多问题负载，不代表任意模型、Prompt 或流量分布。
- Compact `keep_ratio=0.6` 是有损 operating point。当前数据没有证明它适合作为默认
  生产配置；真实服务需要按业务质量要求选择保留率或关闭压实。
- Prefix Cache 位于单个 Engine Process，不跨进程或跨机器共享。
- vLLM/SGLang 比较对齐了模型、KV 字节预算、输入和请求流，但各引擎内部缓存实现不同，
  且外部引擎不暴露等价驻留条目计数。
- 没有把 Uniform、内容寻址或淘汰分数包装成算法创新。可复现的贡献是压缩感知 Prefix
  Cache 的运行时实现、页生命周期、三引擎共用负载和完整的容量—延迟—质量证据。
