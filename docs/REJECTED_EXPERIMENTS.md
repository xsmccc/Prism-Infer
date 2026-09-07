# 未采用方案与历史实验

记录项目探索过的方案及当时依据。由实现缺陷造成的结果不能当作算法结论；下面先更正
视觉剪枝部分，其余条目保留原实验范围。

## 1. 视觉 Token Pruning 与 KV 压实

默认不删除视觉 token，Uniform 和 Attention Top-k 仍作为可选研究配置保留。此前将它们
写成“已被质量实验证伪”不准确：当时 FP8 压实 kernel 存在 int32 地址溢出。

### 实现与动机

在固定 KV 预算下，删除部分视觉 token 可以减少要保存的 KV。选择逻辑位于
`prism_infer/engine/visual_pruning.py`，物理复制和页回收由
`kv_compaction_coordinator.py`、`ops/kv_compaction.py` 及 BlockManager 配合完成。
支持 `visual_compact`、`visual_compact_fp8`、`visual_compact_scaled_fp8`。

### 更正后的证据

2026-08-13 的 int64 地址修复后，MuirBench 全部 85 题中 Dense/Uniform 为 46/85、
47/85；实际删除视觉 token 的同一组 49 题为 27/49、28/49。两种配置有 3 个答案不同。
原始逐题记录见 [cache_pressure_20260813](../artifacts/cache_pressure_20260813/README.md)。

旧的 Uniform 20/49、两种 Attention Top-k 20/49，以及 MVBench 183/252 → 113/252
均来自修复前记录，不能据此断言 Uniform 质量更差或 Attention 没有价值。后两类对照尚
没有修复后重跑结果。DocVQA 的 190 个样本未触发删除，也不能证明剪枝保留了 OCR 精度。

不默认开启剪枝，是因为删除上下文会改变模型计算，而现有修复后质量证据只覆盖很小的
样本集合；不是因为旧的 -14.3pp 已经证明算法不可用。同样，FP8 存储减少 48.44% 是容量
结果，不能写成“FP8 提供无损容量”。

---

## 2. Qwen3-VL-30B-A3B MoE + Pipeline Parallel

**状态：已放弃（2026-08），改动保留在 `moe-30b-wip` 分支，未作为主路径。**

### 动机

扩大模型规模到 30B-A3B，探索 MoE 在无 NVLink 4 卡集群上的可行性。

### 实测（通信探测 + bubble 基准）

- 集群无 IB/NVLink（`NCCL_IB_DISABLE=1` 实测），TP2 30B 的通信开销吃掉计算收益；
- PP bubble 基准（`jobs/pp_bubble_bench.py`）显示 15 分钟作业墙内无法完成
  30B 的 CUDA Graph 捕获（>6.5 分钟起步）；
- EAGLE3 投机解码在 30B + TP2 上实测 acc_len 1.32-1.37（文本），未达预期。

### 放弃原因

15min × 4 卡的受限集群与 30B 的多卡训练/推理规模不匹配；8B 单卡上已形成
完整的缓存/FP8/CUDA Graph 体系，30B 无法产生可写入简历的干净数字。

---

## 3. EAGLE3 投机解码（多模态负载）

**状态：该实验配置未采用为默认推理路径（2026-08-18）。**

### 实测（vLLM 0.25.1 + taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3，TP1，greedy，k=4）

| 负载 | acc_len | 接受率 |
| --- | --- | --- |
| 八图 QA（合成图 + MuirBench 风格问题） | **1.12** | 27.9% |
| 纯文本对照 | 0.98 | 24.4% |
| 参照：30B + SpecForge draft（文本） | 1.32-1.37 | 8.3% |

### 放弃原因

acc_len 远低于 breakeven（约 1.5-1.8）：VQA 短答案分布（数字/颜色/选项）与
draft 训练时的文本续写分布差距大，第 2 位之后的 draft 几乎全部被拒。
在 Prism-Infer 中完整实现 EAGLE3 集成的计划保留（draft runner / verify 走
paged prefill / 回滚 / CUDA Graph），定位为工程能力证明而非主路径加速。
