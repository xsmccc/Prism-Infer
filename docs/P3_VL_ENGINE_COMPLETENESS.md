# P3 VL Engine 完整性与性能基线设计

> 修订日期: 2026-07-16
> 阶段目标: 在 P2 单图 eager strict baseline 上，补齐真实多模态推理框架必须具备的多图、视频、batch 混合、长输出、CUDA Graph decode 和 paged decode kernel 能力。
>
> 历史说明: 本文记录 P3 当时的范围与决策。P7.3 已在 `e7796e9` 补齐
> chunked paged prefill 与 online mixed-VL；本文中的 P3 阶段排除项不代表当前主线
> 仍不支持该能力。

## 范围

P3 不做 KV Cache 压缩策略。P3 的职责是把 Prism-Infer 的 VL baseline 做完整、可验证、可 benchmark:

```text
通用 VL 输入
  -> image/video processor boundary
  -> image/video token span 与 grid 元数据
  -> 3D position ids / rope_delta
  -> Sequence + scheduler batch
  -> ModelRunner prefill/decode
  -> eager reference
  -> CUDA Graph decode
  -> paged decode kernel
  -> long-output quality and benchmark
```

P3 的执行顺序必须是 correctness 优先:

1. P3.0 设计门禁。
2. P3.1 多图输入 correctness。
3. P3.2 视频输入 correctness。
4. P3.3 batch 混合图文 correctness。
5. P3.4 长输出多 token 质量评估。
6. P3.5 VL CUDA Graph decode。
7. P3.6 高性能 paged decode kernel。
8. P3.7 阶段 Review。

## 当前阻断证据

| 能力 | 当前限制 | 证据 |
|---|---|---|
| 多图输入 | P3.1 已解除单请求多图 eager correctness 阻断；P3.3/P3.5 已补齐 mixed batch 与 CUDA Graph decode correctness。 | `prism_infer/engine/vl_inputs.py`, `prism_infer/vision/vision_encoder.py`, `tests/test_full_model_vl_multi_image.py`, `tests/test_llm_vl_generate.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_llm_vl_cuda_graph_decode.py` |
| 视频输入 | P3.2 已解除单请求 synthetic video eager correctness 阻断；P3.3/P3.5 已补齐 mixed batch 与 CUDA Graph decode correctness；真实视频文件读取/采样策略仍未覆盖。 | `prism_infer/engine/vl_inputs.py`, `prism_infer/models/qwen3_vl_position.py`, `prism_infer/models/qwen3_vl.py`, `tests/test_full_model_vl_video.py`, `tests/test_llm_vl_generate.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_llm_vl_cuda_graph_decode.py` |
| batch 混合图文 | P3.3/P3.5 已覆盖 non-prefix eager/Graph；P7.3 已覆盖 online arrival、mixed-VL 与 chunked paged prefill。VL prefix hash显式禁用，避免相同占位 token错误复用不同像素 KV。 | `prism_infer/engine/model_runner.py`, `prism_infer/engine/online.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_llm_online_serving.py` |
| 长输出质量 | P3.4 已覆盖单图/多图/视频 32-token 生成诊断、稳定前缀门禁、teacher-forced logits/ppl 分布和 mixed batch 长输出稳定性；text-only mixed 32-token 分叉已证明为 HF/Prism 共有 batch-size 数值敏感性，视频长输出第 6 token 分叉已定位为 bf16 tie-break。 | `tests/test_llm_vl_long_generate.py`, `tests/test_llm_vl_mixed_batch_generate.py`, `tests/test_vl_logits_distribution.py`, `tests/test_batch_numeric_sensitivity.py` |
| VL CUDA Graph decode | P3.5 已解除阻断；graph replay 支持 `[3,batch]` decode positions，公开 VL 入口允许 `enforce_eager=False`。 | `prism_infer/engine/model_runner.py`, `prism_infer/engine/llm_engine.py`, `tests/test_llm_vl_cuda_graph_decode.py` |
| 高性能 paged decode | P3.6 已接入自实现 Triton paged decode kernel；PyTorch SDPA eager reference 仍保留用于 correctness。 | `prism_infer/ops/paged_decode.py`, `prism_infer/layers/attention.py`, `tests/test_paged_decode_kernel.py` |

## 外部参考证据

本项目核心模型、attention、M-RoPE、KV cache、scheduler 和 kernel 仍必须自实现。以下 HF 源码只作为 processor behavior 和 ground-truth reference:

- HF Qwen3VL processor 返回 `input_ids/attention_mask/pixel_values/pixel_values_videos/image_grid_thw/video_grid_thw`，源码位置: `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py:146-155`。
- HF processor 对多图占位 token 的展开逻辑: `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py:186-194`。
- HF processor 对视频 token 按 timestamp 和 frame placeholder 展开: `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py:196-234`。
- HF `get_rope_index` 支持 `image_grid_thw` 与 `video_grid_thw`，并在视频路径中先 `repeat_interleave(video_grid_thw, video_grid_thw[:, 0])` 后把 T 置为 1，源码位置: `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:916-1033`。
- HF decode 使用 prefill 阶段缓存的 `rope_deltas` 生成后续 decode position ids，源码位置: `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1177-1221`。

## 设计决策

### D1: 用通用 VLInputs 替代 SingleImageInputs 语义

选择:

- 新增或重命名为 `VLInputs`，承载 image/video 两类 payload。
- 字段应至少包含:
  - `input_ids: [1, seqlen]`
  - `attention_mask: [1, seqlen]`
  - `pixel_values: [num_image_patches, patch_dim] | None`
  - `image_grid_thw: [num_images, 3] | None`
  - `pixel_values_videos: [num_video_patches, patch_dim] | None`
  - `video_grid_thw: [num_videos, 3] | None`
  - `image_token_id/video_token_id`
  - `image_token_count/video_token_count`
  - `expected_image_tokens/expected_video_tokens`
  - `prompt_text`
- 保留 `prepare_single_image_inputs` 兼容旧测试，但内部可委托到通用 `prepare_vl_inputs`。

拒绝:

- 不继续把多图硬塞进 `SingleImageInputs` 名称。名称会误导后续 batch/video 实现，容易隐藏 unsupported state。
- 不把 HF model 作为运行时 wrapper；HF processor 仍只作为非核心预处理工具。

风险:

- 视频 processor 字段和 metadata 对 transformers 版本敏感。实现前必须用本地 processor 输出写测试，不能凭记忆实现。

### D2: image/video token 数必须由 grid 和 merge_size 推导

选择:

- 图片 token 数:

```text
sum(image_grid_thw.prod(dim=1) // image_merge_size**2)
```

- 视频 token 数第一层按 processor 输出校验；rope index 内部按 HF 行为对 `video_grid_thw` 做 per-frame 展开:

```text
expanded_video_grid = repeat_interleave(video_grid_thw, video_grid_thw[:, 0])
expanded_video_grid[:, 0] = 1
```

拒绝:

- 不写死 `196`、`784`、`T=1` 等常量。
- 不用 image token 规则假装视频规则。视频有 timestamp 展开，不能被当成一张大图。

风险:

- 视频 prompt 中 timestamp token 会改变文本长度和 rope_delta；必须和 HF `get_rope_index` exact match 后再进入模型 forward。

### D3: Position ids helper 同时支持 image/video/mixed spans

选择:

- 扩展 `prism_infer/models/qwen3_vl_position.py`:
  - `image_grid_thw: [num_images,3] | None`
  - `video_grid_thw: [num_videos,3] | None`
  - batch 内每条请求可以有不同数量 image/video spans。
- 按 token 顺序处理 image/video span，而不是先处理所有 image 再处理 video。
- 输出仍为:
  - prefill `position_ids: [3, batch, seqlen]`
  - `rope_delta: [batch, 1]`
  - decode `position_ids: [3, batch]` 或等价 `[3, num_decode_tokens]` flatten 形态。

拒绝:

- 不在视频 token 出现时继续报错。
- 不用 text-only 1D positions 替代 VL 3D positions。

风险:

- 当前 helper 的 `image_index` 是整个 batch 共享计数；mixed batch 时必须确保 image/video grid 行按 batch 内 token span 顺序消费，并有越界和未消费校验。

### D4: Batch 混合先做 correctness，再优化调度效率

选择:

- 放开 `ModelRunner.prepare_prefill/prepare_decode` 对 VL 单 sequence 的限制。
- prefill flatten 时允许:
  - text-only 使用 1D positions。
  - VL 使用 `[3, seqlen]` positions。
  - 统一转成模型可消费的 `[3, total_tokens]` 或兼容形态。
- pixel payload 可以先按 batch 中 VL 请求顺序 concat，并记录 grid 行顺序。

拒绝:

- 不先做 prefix-cache/chunked prefill 与 VL 的复杂组合。P3.3 先保证 non-prefix mixed batch correctness，unsupported 组合显式报错。
- 不为了 batch 支持牺牲单请求 strict alignment。

后续状态: P7.3 已移除上述 P3-era early gate并实现 Q<K paged gather+SDPA
correctness路径。视觉 payload按 atomic region分块；VL prefix hash因 token id不包含像素
语义而保持禁用。这是后续阶段实现，不改变 P3.3 当时的验收范围。

风险:

- KV slot mapping、block table 和 `cu_seqlens` 是 batch correctness 的高风险点；测试必须比较 mixed batch 与单请求独立运行。

### D5: 长输出质量不能只看首 token

选择:

- 对 greedy 模式做 `max_tokens=32` 生成诊断和稳定前缀 token 对齐。
- 对每个生成步记录至少:
  - logits shape
  - max diff
  - mean diff
  - mean/std
  - PASS/FAIL
- 对采样模式只比较分布指标或 ppl，不要求随机输出文本逐字一致。

拒绝:

- 不把 1-token greedy PASS 外推为长输出质量 PASS。
- 不用“文本看起来合理”替代数值验证。

风险:

- 长输出中首个分叉可能来自极小 logits 差异放大或 bf16 tie-break；测试必须记录首个分叉 step，并在必要时记录该步候选 logits，便于定位。

### D6: CUDA Graph decode 必须先对齐 eager reference

选择:

- 保留 eager VL decode 作为 reference。
- graph replay 需要支持 3D position ids 和 mixed batch 的 rope_delta。
- `enforce_eager=False` 的公开 VL 入口只有在 graph correctness PASS 后才能放开。

拒绝:

- 不在 graph path 未验证时移除 `generate_vl` 的 eager 限制。
- 不把纯文本 CUDA Graph PASS 当成 VL CUDA Graph PASS。

风险:

- 当前 graph `positions` 占位是 `[max_bs]`；VL decode 需要 `[3, max_bs]` 或统一 flatten 形态，capture/replay 地址和 shape 都要稳定。

### D7: Paged decode kernel 以 eager fallback 为 reference

选择:

- 当前 `_forward_decode_eager` 作为 correctness reference。
- 新 kernel 必须显式接收 q、paged k/v cache、block_tables、context_lens、num_heads、num_kv_heads、scale 等必要参数。
- correctness PASS 后再 benchmark。

拒绝:

- 不在 kernel unsupported shape 时 silent fallback 到 eager 并报告 kernel PASS。
- 不在没有 shape/max diff/mean diff 的情况下报告性能收益。

风险:

- Qwen3-VL 使用 GQA，`num_heads != num_kv_heads` 时 repeat/group 语义必须与 eager reference 一致。

## 测试矩阵

| 子任务 | 最小输入 | correctness reference | 主要输出 |
|---|---|---|---|
| P3.1 多图 | 1 prompt + 2 images | HF processor/get_rope_index/full logits/generate | `image_grid_thw=[2,3]`, logits max diff, greedy tokens |
| P3.2 视频 | 1 prompt + synthetic video | HF processor/get_rope_index/full logits/generate | `video_grid_thw`, video token count, logits max diff |
| P3.3 mixed batch | text-only + single-image + multi-image + video | 单请求独立运行 | per-seq logits/token ids, KV max diff |
| P3.4 长输出 | max_tokens 32 | HF greedy/generation logits | 稳定前缀、首个分叉点、logits/ppl 分布 |
| P3.5 CUDA Graph | single/multi/mixed decode | eager VL decode | graph logits/token ids diff, latency |
| P3.6 paged kernel | batch 1/2/4/8, context 256/1024/4096 | `_forward_decode_eager` | q/k/v/cache shape, max diff, token/s |

当前状态:

- P3.1 多图已 PASS:
  - processor: `input_ids=[1,408]`, `pixel_values=[1568,1536]`, `image_grid_thw=[2,3]`, image tokens `392 / 392`。
  - rope index: `position_ids=[3,1,408]`, `rope_delta=[1,1]`, position/delta max diff `0.000000e+00`。
  - full logits: HF/Prism shape `[1,151936]`, mean/std 完全一致，max diff `0.000000e+00`, mean diff `0.000000e+00`。
  - `LLM.generate_vl`: HF multi-image token ids `[785]`，Prism token ids `[785]`。
- P3.1 关键问题记录为 `docs/ISSUE_LOG.md` 中的 `P3-001`。
- P3.2 视频已 PASS:
  - processor: `input_ids=[1,420]`, `pixel_values_videos=[1568,1536]`, `video_grid_thw=[[2,28,28]]`, video tokens `392 / 392`。
  - rope index: `position_ids=[3,1,420]`, `rope_delta=[1,1]`, position/delta max diff `0.000000e+00`。
  - full logits: HF/Prism shape `[1,151936]`, mean/std 完全一致，max diff `0.000000e+00`, mean diff `0.000000e+00`。
  - `LLM.generate_video`: HF video token ids `[785]`，Prism token ids `[785]`。
- P3.2 关键问题/实现记录为 `docs/ISSUE_LOG.md` 中的 `P3-002`。
- P3.3 mixed batch 已 PASS:
  - ModelRunner mixed prefill: `input_ids=[1043]`, `position_ids=[3,1043]`, `pixel_values=[2352,1536]`, `image_grid_thw=[3,3]`, `pixel_values_videos=[1568,1536]`, `video_grid_thw=[1,3]`。
  - ModelRunner mixed decode: `input_ids=[3]`, `position_ids=[3,3]`, `context_lens=[6,211,421]`。
  - `LLM.generate_mixed`: text/single-image/multi-image/video mixed batch token ids `[[11], [785], [785], [785]]`，与 fresh 单请求独立运行一致。
- P3.3 关键问题/实现记录为 `docs/ISSUE_LOG.md` 中的 `P3-003`。
- P3.4 长输出已 PASS:
  - single-image/multi-image `max_tokens=32` 生成中 `prefix@8/16` 与 HF 一致；video `prefix@5` 与 HF 一致，首个分叉发生在第 6 token 的 bf16 tie-break。
  - single-image/multi-image/video teacher-forced logits shape `[1,32,151936]`，logits mean diff 约 `4.7e-3` 到 `5.3e-3`，ppl diff `< 0.01`，分布级 PASS。
  - mixed batch VL rows 长输出保持稳定前缀；multi-image/video 当前与 fresh 单请求独立运行一致，single-image 首个分叉在 token 28。
  - mixed batch text-only row 32-token 分叉来自 bf16 batch-size 数值敏感性；HF 与 Prism duplicate batch max diff 同量级，argmax 一致。
- P3.4 关键问题/实现记录为 `docs/ISSUE_LOG.md` 中的 `P3-004`。
- P3.5 CUDA Graph decode 已 PASS:
  - `tests/test_model_runner_vl_cudagraph.py`: text `[batch]` 和 VL `[3,batch]` decode positions 均规范为 `[3,batch]`；非标准 `max_num_seqs` graph 档位覆盖 `max_bs`。
  - single-image/multi-image/video `max_tokens=2` graph token ids 与 eager 完全一致。
  - mixed batch=3 graph token ids 与 eager 完全一致: `[[11, 358], [785, 1378], [785, 2766]]`。
  - 代表性 benchmark: RTX 5090，commit `45edd3a`，mixed，`max_tokens=8`，warmup=2，repeat=5；eager decode median `31.5488ms`，graph decode median `16.4468ms`，correctness PASS。
- P3.6 paged decode kernel 已 PASS 当前基线:
  - 新增 `prism_infer/ops/paged_decode.py`，显式接收 q、paged k/v cache、block_tables、context_lens、GQA group 和 scale。
  - correctness: small GQA max diff `3.906250e-03`；Qwen shape max diff `7.812500e-03`，均低于 bf16 跨实现门槛 `1e-2`。
  - benchmark: batch `1,2,4,8` × context `256,1024,4096` 全部 correctness PASS。
  - 性能风险: batch=1/context=4096 kernel median `0.2834ms` 慢于 reference `0.2314ms`，后续 P6 需优化长上下文单 batch kernel。
- P3.7 阶段 Review 已 PASS:
  - `compileall prism_infer tests benchmarks`: PASS。
  - `git diff --check`: PASS。
  - P1/P2/P3 grouped regression: `49 passed in 356.34s`。
  - 纯文本、单图、多图、视频 full logits 均 strict PASS，max diff 和 mean diff 均为 `0.000000e+00`。
- P3 当前门禁已完成；下一阶段进入 P4 KV Cache 分析。

P3.1 关键修复:

- HF `Qwen3VLVisionModel.forward` 使用 `cu_seqlens` 将多张图按图片分段做 vision attention，证据为 `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:727-735`。
- HF eager vision attention 在非 FA2 路径中按 `cu_seqlens` split 后逐段计算，证据为 `/data/Prism-Infer/.venv-local/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:223-248`。
- Prism-Infer 修复为 `VisionEncoder.forward` 构造 `cu_seqlens`，`ViTAttention.forward` 多图时逐段 SDPA，避免不同图片 patch 之间跨图 attention。

## Benchmark 约束

所有 P3 benchmark 必须输出:

- commit hash。
- GPU 型号、CUDA、torch、transformers。
- dtype、max model len、block size、batch size、context len、image/video token 数。
- warmup 次数、repeat 次数。
- `torch.cuda.synchronize()` timing 边界。
- GPU memory allocated/reserved/peak。
- latency median、p90、min、max。
- throughput 或 token/s。

与 vLLM/SGLang 对比前必须另建对比设计文档，记录对方版本或 commit、启动参数、调度参数、显存限制、输入集合和采样配置。没有这些证据时，不声明超越。

## 问题记录规则

P3 每解决一个真实 bug 或设计阻断，都要更新 `docs/ISSUE_LOG.md`。记录必须包含:

- 问题编号，如 `P3-001`。
- 触发命令和失败输出摘要。
- 根因定位证据，包含文件路径/行号或测试输出。
- 修复方案和拒绝的替代方案。
- 验证命令、shape、max diff、mean/std、PASS/FAIL。
- 剩余风险。

## P3 完成时的下一步（历史）

当前应执行 P4.0 KV Cache 分析设计门禁:

1. 设计 trace schema，记录模型配置、输入类型、token 区间、层号、head、KV/attention 统计量。
2. 设计 trace 开关，默认关闭，开启后必须验证 greedy 输出不变。
3. 先覆盖 single-image、multi-image/video、mixed batch 中至少两类输入。
4. P4 只做观测与分析，不提前实现压缩策略。
