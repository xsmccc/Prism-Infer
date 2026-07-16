# Prism-Infer Claim Ledger

> 冻结基线: `p6.12-content-aware-kv` (`c970c61`)
> 更新日期: 2026-07-16

本表区分“已实现”“已验证”和“性能占优”。README、简历和面试中的数字必须能
追溯到本表及对应 raw evidence。

## 可以使用的结论

| 结论 | 范围 | 证据 |
|---|---|---|
| Prism 自实现 Qwen3-VL text/vision/M-RoPE/DeepStack/engine 主路径 | Qwen3-VL-8B-Instruct | `VERIFICATION.md` P1-P3 |
| visual KV 是真实 physical compaction，不只是逻辑 mask | BF16/FP8 paged KV；prefill 后 compact、page 回收、decode append | P6.4 tests 与 `PERFORMANCE_REPORT.md` |
| logical M-RoPE position 与 physical KV position 分离 | compact decode | layout/append/mixed/swap focused regression |
| content-aware last-layer scorer 通过当前 reference task gate | 7 张固定 COCO 图片、35 captions、output32、keep=0.5、BF16 | token-F1 `0.321635 -> 0.318347`，drop `0.003288`；ROUGE-L `0.289116 -> 0.285406`，drop `0.003710` |
| 质量合格策略减少物理 KV | 同一 7-image gate | physical token ratio `0.536x`，active prompt bytes ratio `0.571x` |
| 压缩 CUDA Graph 路径有效 | RTX 5090，offline decode，batch1-8 | eager/Graph token exact；decode speedup约 `1.76x-1.94x`，见 P6.11 |
| 当前质量合格压缩的短 workload 性能收益很小 | COCO batch4/output32 | decode-step `1.021x`，engine output throughput `1.013x`，E2E `1.005x` |
| P6.12 后全量回归通过 | 单卡环境 | `238 passed, 6 skipped in 232.90s` |

## 必须带限制的结论

| 现象 | 必须同时说明 |
|---|---|
| uniform/FP8 组合曾观察到 `4.016x` peak running capacity | uniform quality FAIL；FP8 quality 未通过；不是 online throughput |
| active prompt bytes 降至 `0.571x` | 不是整个模型/GPU peak memory 降至 `0.571x` |
| CUDA Graph 提升约 1.8 倍 | 是 Prism internal eager→Graph，不是对 vLLM speedup |
| reference token-F1/ROUGE-L drop 小于 0.004 | 不是标准 COCO CIDEr/SPICE，也不是通用 VQA accuracy |
| external eager baseline 比 Prism eager 快约 2 倍 | 仅为 P6 diagnostic matched eager；P7 重新比较双方 Graph |

## 当前禁止的结论

- “Prism 全面超过 vLLM/SGLang”。
- “KV 压缩让整体 GPU 显存减半”。
- “标准 COCO accuracy 下降小于 1%”。
- “FP8 KV 已通过质量门禁”。
- “offline batch tok/s 等价于 online serving throughput/goodput”。
- “TP2 已验证”或“多卡可扩展”；当前机器只有一张可见 RTX 5090。
- “已实现 megakernel/PD 分离/投机解码”。

## P7.1 待填结论

P7.1 只在 schema-v2 全部 comparability checks 通过后填写：

- `diagnostic_matched`: Prism eager vs vLLM eager。
- `best_stable`: Prism Graph vs vLLM effective Graph。
- Prism off 与 quality-qualified `visual_compact_graph` 必须同时出现。
- 当前 P7.1 仍是 offline closed-loop，不形成 online SLO goodput claim。
