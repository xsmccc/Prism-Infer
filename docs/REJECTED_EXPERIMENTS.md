# 已探索并被证伪的方向（REJECTED EXPERIMENTS）

本文档记录 Prism-Infer 探索过、实测后放弃的技术方向及放弃依据。保留这些记录
是为了让后续读者（以及面试官）不用重复踩坑：每一项都有动机、实现、测量和结论。

## 1. 视觉 token 剪枝 / KV 压实（Visual Pruning / Compaction）

**状态：已放弃（2026-08）。Dense Scaled-FP8 是唯一推荐路径。**

### 动机

4 GiB KV 预算装不下完整工作集的视觉前缀（pressure 工作集需要 312 页，预算只有
220 页）。剪掉一部分视觉 token 的 KV（"压实"）可以让更多媒体组驻留，提高命中率、
降低 TTFT。

### 实现

- `prism_infer/engine/visual_pruning.py`：query-agnostic Uniform 采样与
  Attention Top-k 两种选择策略；
- `prism_infer/engine/compression.py` / `kv_compaction_coordinator.py`：物理压实
  （copy 保留行、回收页）、压缩后页共享、CoW/tail-clone 全链路；
- 配置层 `compression_mode` 支持 `visual_compact` / `visual_compact_fp8` /
  `visual_compact_scaled_fp8`。

### 实测（质量协议，49 个 MuirBench 样本 + 252 个 MVBench 视频样本）

| 配置 | MuirBench 准确率 | 说明 |
| --- | --- | --- |
| 参考（不剪枝） | **27/49 (55.1%)** | |
| Uniform 复用（第一题决定剪枝集，跨问题复用） | 20/49 (40.8%) | -14.3pp |
| Attention Top-k（每题独立） | 20/49 (40.8%) | 与 Uniform 无差异 |
| Attention Top-k（复用第一题选择） | 20/49 (40.8%) | 不优于 Uniform |

- 视频：MVBench 252 样本实际删除 20,064 个视觉 token（约 183→113 可保留问题数，
  见 working_set_quality.csv）。
- DocVQA：受 768-token 最低保留量限制，实际未发生删除（0 样本被压缩）。
- 容量侧：压缩路径运行中 Prefix 页数 -29.92%，可驻留媒体组 +48.15%。

### 放弃原因

1. **质量损失不可接受**：-14.3pp 意味着答案直接错掉。前缀缓存的立身之本是
   "无损复用"（相同输入 → 相同输出）；剪枝把缓存变成了有损压缩，违背设计前提。
2. **Attention 选择没有更好的可复用方案**：逐题 Top-k 无法跨问题复用（每题
   attention 不同），复用第一题的选择也不优于 Uniform——说明这个负载下不存在
   "既压缩又可复用且不损质量"的简单选择策略。
3. **FP8 已提供无损容量**：Scaled-FP8 每 token 存储 -48.44%，容量 ×1.95，是
   来自量化而不是删 token 的收益。

### 遗产

压实/CoW/tail-clone 的代码保留在仓库中（`compression_mode` 默认 `off`），
作为显式对照配置与测试资产；Dense FP8 Prefix 成为唯一默认路径。

---

## 2. Qwen3-VL-30B-A3B MoE + Pipeline Parallel

**状态：已放弃（2026-08），改动归档在 `moe-30b-wip` 分支，永不合并。**

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

**状态：闸门实测后判定不划算（2026-08-18），集成实现保留为工程能力证明。**

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
