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
| P7.1 外部比较协议可自动拒绝不公平 cell | schema-v2 offline closed-loop | 两条 profile 共 20 rows 全部通过 model/GPU/KV/execution/clean-state gates |
| Prism Graph 明显缩小 eager 开销，但当前仍慢于 vLLM Graph | RTX 5090、固定五类 workload、output32 | quality-qualified compact Graph TPOT 为 vLLM `1.65x-1.78x`；不能声称已超过 |
| content-aware compaction 对当前短/中 visual context只有小幅 TPOT收益 | 同一 P7.1 matrix | compact 相对 Prism off Graph约改善 `1.5%-3.0%` |

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

## P7.1 当前结论

- `diagnostic_matched`: Prism eager TPOT约为 vLLM eager 的 `1.91x-1.97x`。
- `best_stable`: Prism off Graph约为 vLLM Graph 的 `1.69x-1.83x`；quality-qualified compact Graph约为 `1.65x-1.78x`。
- 双方 E2E throughput 当前也是 vLLM 更高，但部分 Prism offline TTFT存在双峰，E2E不作为压缩收益归因。
- 这是 offline closed-loop，不形成 online SLO goodput claim；P7.3 需要把 KV page容量转化为在线 admission/goodput 后重新比较。
