# P7-004: 物理 token 减少不保证 page/bytes 回收

- 状态: `DOCUMENTED_LIMITATION`
- 首次正式证据: `b17f933`
- 硬件/软件: RTX 5090，Qwen3-VL-8B，BF16，Prism CUDA Graph
- KV block size: 256 tokens
- KV pool: 603,979,776 bytes
- 影响: physical KV capacity、admission control 与显存 claim

## 现象

single-image keep=0.5 将 physical prompt token 从 `210` 降到 `112`，但
`active_prompt_bytes` 仍为 `37,748,736`，没有下降。multi-image 从 `408` 降到
`212` 时，active bytes 才从 `75,497,472` 降到 `37,748,736`。

## 如何发现

P7.1 同时记录三个不同口径：

- logical/physical token count。
- active/dense block count。
- active/dense prompt bytes。

如果只看 token ratio，single-image 会被错误写成节省约 47% KV 显存；block/bytes
证据立即暴露该结论不成立。

## 复现命令

同一命令同时产生 dense 与 compact 的 token/page/bytes 记录；将 `--case` 改为
`multi_image_2x448` 可复现跨 page boundary 后的实际回收：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-local/bin/python benchmarks/bench_system.py \
  --model /data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case single_image_448 \
  --modes off_graph,visual_compact_graph \
  --max-tokens 32 --warmup 2 --repeat 5 \
  --max-model-len 1280 --max-num-batched-tokens 2048 \
  --num-kvcache-blocks 16 --kvcache-block-size 256 \
  --disable-prefix-caching \
  --visual-pruning-keep-ratio 0.5 \
  --visual-pruning-strategy attention \
  --visual-pruning-attention-last-n-layers 1 \
  --output data/p7_external/prism_page_granularity_repro.jsonl
```

## 根因

Paged KV 的分配粒度是 256 tokens：

```text
ceil(210 / 256) = 1 page
ceil(112 / 256) = 1 page

ceil(408 / 256) = 2 pages
ceil(212 / 256) = 1 page
```

compaction 会把 retained KV 搬到连续物理槽并释放完整尾页，但不能释放仍含任意
有效 token 的首尾 page。

## 为什么机制仍有效

physical token 变短会减少 decode attention 读取量，因此 single-image Graph
TPOT仍从 `17.932` 降到 `17.671 ms`。但 admission capacity 只能按实际回收的
page 计算，不能按 token ratio 线性外推。

## 解决方案与工程结论

- scheduler/admission 使用 free page 和 active bytes，不使用 token ratio。
- benchmark 同时报告 token/page/bytes。
- 后续可把 block size 作为独立实验轴，但改变 page size 会影响 kernel、元数据和
  fragmentation，不能只为一个样例调小后声称普遍更优。

## 验证

P7.1 五类 workload 都同时记录了 physical tokens 与 active bytes：single image
保持 `1→1` page；two images 为 `2→1` pages；两个 fidelity batch 分别为
`7→4` 和 `6→3` pages。对应正式 TPOT 与 page/bytes 记录均来自 clean
`b17f933`。

## 被拒绝的方法

- 用 `retained_tokens / dense_tokens` 直接充当显存 ratio：忽略 allocator 粒度。
- 为 single-image 直接把 block size 调小并宣称修复：会同时改变 attention kernel、
  block table 开销和碎片率，必须作为完整矩阵单独评估。
- 用 process peak allocated 判断 compaction 失效：固定 KV pool 和模型权重不会因
  active tail page 回收而缩小，peak 与 active capacity 回答的是不同问题。

## 剩余限制

- 当前只验证 block size 256；尚未完成 64/128/256 的 latency、元数据和碎片率消融。
- offline 单批次没有证明释放 page 会提升在线 admission/goodput；该闭环属于 P7.3。
- 多请求长期运行下的 fragmentation、取消和 preemption 回收仍需单独验证。

## 面试表达

> 我发现 KV token 压缩率不是显存压缩率，因为 allocator 以 page 为粒度。一个
> 210→112 token 的请求仍占同一页；只有跨过 256-token 边界才真正释放显存。
> 所以调度器基于 physical pages 做 admission，报告也同时给 token 和 bytes。
