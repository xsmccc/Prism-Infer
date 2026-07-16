# P7.0/P7.1 冻结与外部基线设计

> 日期: 2026-07-16
> 状态: implementation/preflight
> 硬件: 单张 NVIDIA GeForce RTX 5090 32GB

## 1. 阶段目标

P7.0 冻结 P6.12 的代码和可发布 claim。P7.1 在同一 deterministic workload、
显存预算和测量协议下重新比较 Prism 与 vLLM，为后续 online scheduler、
CUDA Graph、Inductor 和 Blackwell kernel 调优建立起点。

P7.1 仍是 offline closed-loop baseline，不提供 request arrival、queueing、p99
SLO 或 online goodput。它回答“单轮固定 batch 的执行差距在哪里”，不是“谁的
在线服务能力更强”。

## 2. P7.0 冻结合同

- P6.12 freeze commit: `c970c61`。
- annotated tag: `p6.12-content-aware-kv`。
- 允许和禁止的表述见 `docs/CLAIMS.md`。
- P6.2-B hardware counter、P6.8 TP2 和 FP8 quality 继续保持未完成。
- P7 benchmark harness 必须先提交，再在 clean commit 上生成 formal records。

## 3. 两条对比 profile

### 3.1 `diagnostic_matched`

双方禁用 CUDA Graph：

```text
Prism off_eager / visual_compact
vLLM --enforce-eager, chunked prefill off, async scheduling off
```

用途是保留 P6 的问题归因：模型执行、attention backend、kernel fusion 和
framework overhead 的总差距。它不是双方最佳性能。

### 3.2 `best_stable`

双方使用各自已经验证的稳定 Graph 路径：

```text
Prism off_graph / visual_compact_graph
vLLM effective cudagraph mode != NONE, stable chunked/async defaults enabled
```

vLLM 必须记录实际 `cudagraph_mode` 和 `compilation_mode`，不能只根据命令行
“未传 enforce-eager”推断 Graph 生效。

## 4. 自动 comparability gates

schema-v2 只有在以下条件全部成立时输出性能 ratio：

- comparison profile 与双方 effective execution 匹配。
- manifest/case/request count/max output 相同。
- prompt token 总数相同。
- 模型 `config.json` SHA256、dtype、TP 相同。
- max model len、max batched tokens、max sequences 相同。
- KV pool bytes、block size、prefix cache 设置相同。
- temperature 0、ignore EOS、max tokens相同。
- preprocessing/output decoding 的 E2E scope 相同。
- warmup/repeat 和 CUDA synchronize 要求相同。
- GPU UUID 相同。
- Prism、benchmark harness 和 external source 均为 clean fixed commit。

Torch/CUDA/framework版本和 attention backend允许不同，因为它们属于被比较的
系统实现，但必须完整记录。

## 5. 固定环境与配置

```text
model: Qwen3-VL-8B-Instruct / 0c351dd...
dtype: BF16
TP: 1
max_model_len: 1280
max_num_batched_tokens: 2048
block_size: 256
KV pool: 603,979,776 bytes
prefix cache: off
MM processor cache: off (vLLM)
sampling: temperature=0, ignore_eos=true
output: 32
warmup/repeat: 2/5
```

Prism content-aware mode固定 `keep=0.5`、attention strategy、last one decoder
layer。vLLM 使用 `FLASH_ATTN` attention 和当前 Blackwell 环境中可运行的 PyTorch
native sampler。

## 6. 最小 formal matrix

| Manifest | Case | 目的 |
|---|---|---|
| synthetic | single image 448 | batch1/短 visual context |
| synthetic | two images 448 | 更高 visual KV 占比 |
| real | COCO single image | 固定真实输入 |
| real | fidelity batch A (4) | Graph batch、质量集前半 |
| real | fidelity batch B (3) | 非 2 次幂 Graph bucket、质量集后半 |

`diagnostic_matched` 至少覆盖前三项；`best_stable` 覆盖全部五项。video/mixed
在 vLLM processor timestamp semantics 与 Prism prompt token 不同，保留 raw
record但不能进入 ratio，除非先统一输入语义。

## 7. 指标

- engine TTFT。
- decode TPOT/step median、p90、p99、min、max。
- E2E latency 和 output tokens/s。
- torch allocator peak memory。
- stable prefix/token exact，仅作为数值路径审计，不把 vLLM输出作为质量 reference。
- Prism physical prompt tokens、active prompt bytes。
- effective CUDA Graph/compile configuration。

## 8. 执行与汇总

formal run 必须一 case/backend 一个 fresh external process。Prism runner 每个
case/mode 重建 model，释放后再进入下一 mode。原始 stdout/stderr 与 JSON/JSONL
保存在忽略跟踪的 `data/p7_external/`，报告只引用文件名和自动汇总结果。

汇总入口：

```bash
.venv-local/bin/python scripts/summarize_p7_external.py \
  --comparison-profile best_stable \
  --prism <prism-jsonl...> \
  --external <vllm-json...> \
  --prism-modes off_graph visual_compact_graph \
  --json-output data/p7_external/best_stable_summary.json \
  --markdown-output data/p7_external/best_stable_summary.md
```

## 9. P7.1 出口标准

- schema/summary focused tests PASS。
- P6 schema-v1 历史记录继续兼容。
- P7 harness 位于 clean pushed commit。
- 两条 profile 的最小 matrix完成，失败和 warning 保留。
- 自动汇总没有 non-comparable cell；若有则不得输出该 cell ratio。
- `PERFORMANCE_REPORT.md`、`VERIFICATION.md`、ROADMAP 和 issue 档案同步。
- 全量 regression 不退化。

## 10. 后续 profiling 触发条件

P7.1 只建立差距。如果 best-stable 下 Prism 仍落后，则对相同 single/multi-image
Graph workload采集 Nsight Systems，先分离 Graph 内 GPU time和 Graph 外固定开销，
再决定 Inductor/TK 工作；不凭经验预选 kernel。
