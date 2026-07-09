# 2026-07-08 外部路线评估对照审查

> 来源: 用户提供的“Prism-Infer 技术路线与项目规划”附件。
> 目的: 对照当前仓库源码、测试和验证文档，采纳可证实的问题，修正过期判断，并阻止未验证路线收益进入项目事实口径。

## 审查原则

- 附件中的外部论文、vLLM/SGLang/LMSys 结论和性能数字，本轮未逐项查源，不能作为项目 claim 使用。
- 本轮只修复可由当前仓库源码和测试复现的 correctness / hardening 问题。
- FP8 KV、VScan、PoRe、DeepStack-aware pruning、M-RoPE block compaction 仍属于候选设计或后续路线；未实现前不得写成已完成能力或已验证收益。

## 采纳并修复

| 编号 | 结论 | 本轮处理 | 证据 |
|---|---|---|---|
| B2 | `Sequence.__setstate__` 丢失 `temperature/max_tokens/ignore_eos` 是真实 bug。 | `Sequence.__getstate__/__setstate__` 保存和恢复 sampling 参数。 | `tests/test_sequence_multimodal.py::test_decode_sequence_roundtrip_preserves_sampling_params` |
| P1-2 | `swap_in` 依赖 decode 反序列化对象完整 `token_ids` 是真实风险。 | `swap_out` 保存每个 CPU block 的 hash 与满块 token 副本；`swap_in` 使用 metadata 恢复 prefix-cache index，不再调用 `seq.block(i)`。 | `tests/test_kv_engine_hardening.py::test_block_manager_swap_in_restores_hash_from_metadata_after_decode_pickle` |
| B3 | `ModelRunner.run()` 异常路径不清理 `Context` 是真实 bug。 | 将推理主体包进 `try/finally`，异常时也 `reset_context()`；chunked prefill 临时截断状态也在 finally 恢复。 | `tests/test_model_runner_context_reset.py` |
| B4 | `scheduler.py` 使用 `assert scheduled_seqs` 会在 `python -O` 下失效。 | 改成显式 `RuntimeError`。 | `tests/test_scheduler_swap_tables.py::test_scheduler_empty_decode_raises_runtime_error` |
| C-4 | trace 关闭时仍调用 `record_attention_layer(...)` 有不必要参数构造开销。 | `Attention.forward()` 先检查 `is_trace_enabled()`，关闭时不调用 trace 记录函数。 | `tests/test_compression_off.py` 和 P4/P5 focused 回归通过 |
| C-7 | paged decode kernel 只检查 max diff，mean diff 门槛偏弱。 | 增加 `mean diff < 1e-3` 断言。 | `tests/test_paged_decode_kernel.py` 更新 |

## 已覆盖或证伪

| 编号 | 附件判断 | 当前复核结论 |
|---|---|---|
| P1-1 | `Sequence.block_size` 类变量仍有多实例共享风险。 | 已有 `Sequence.set_block_size()` 与 BlockManager mismatch gate；本轮进一步给每个 `Sequence` 保存实例级 `block_size` 快照，避免构造后被后续 Config 改写影响。 |
| B1 | engine flatten VL 路径可能没有 DeepStack 注入。 | 当前源码中 `ModelRunner._forward_model()` 会传 `pixel_values/image_grid_thw`，`Qwen3VLModel.forward()` 会构造 `visual_pos_masks/deepstack_visual_embeds` 并传给 language model；本轮新增 engine-style flatten 轻量测试覆盖该路径。 |
| P1-3 | KV Trace prefill 路径零测试。 | 当前 P4 代码和测试已覆盖 prefill trace on/off；附件中也标注过期。 |
| P1-4 | KV Trace 缺少 attention entropy。 | 当前 `kv_trace.py` 已实现 entropy，并进入 P5.1 scoring 输入；附件中也标注过期。 |
| `benchmarks/` 不存在 | 文档引用 benchmarks 目录但目录不存在。 | 当前仓库已有 `benchmarks/bench_paged_decode.py` 和 `benchmarks/bench_vl_cudagraph_decode.py`。该问题过期。 |
| `CLAUDE.md` 缺少 `ops/analysis` | 目录结构未更新。 | 当前 `CLAUDE.md` 已列出 `ops/` 与 `analysis/`。该问题过期。 |
| `P4_KV_TRACE_DESIGN.md` 状态过期。 | 本轮已将状态改为 `Verified`。 |

## 暂缓项

| 编号 | 暂缓原因 |
|---|---|
| C-1 `_cfg_get` 抽取 | 纯重构，不影响当前 P5 correctness；会扩大 diff 和回归范围。 |
| C-2 `config.py` 对 engine compression 的依赖 | 架构清理项；P5.0 off guard 当前验证通过，留到 compression 模块稳定后再拆。 |
| C-3 端口 2333 / SHM 1MB 参数化 | TP/多进程配置质量项；当前任务聚焦 P5 readiness correctness。 |
| C-5 decode trace 开销开关 | 需要设计 trace config schema 变更；可放到 P4/P5 trace 性能优化。 |
| C-6 visual importance aggregate 合并 | 纯内部重构，当前 P5.1 focused tests 已覆盖行为。 |

## P5 路线修正

附件提出的 FP8 KV、VScan+PoRe、M-RoPE block compaction 和竞品对比可以作为候选路线，但当前仓库事实是:

- 已实现: P5.0 compression-off baseline，P5.1 offline visual-token importance scoring。
- 未实现: active pruning、runtime retention mask、physical KV compaction、FP8 KV cache、mixed precision attention。
- 未验证: 任意 compression-on 的压缩率、质量退化、显存收益、latency/throughput 收益。

因此 P5.2 的最小可行目标仍应先证明一个可回退的 visual-token pruning/retention 策略:

1. 保留 FP off reference。
2. 基于 P5.1 score 生成可审计 retention decision。
3. 明确处理 block size 与 M-RoPE position 的限制。
4. 输出 token 保留率、质量退化和性能/显存数据。
5. 未支持的物理 compaction / FP8 / VScan / PoRe 继续显式失败或标为未实现。

## 本轮验证

```bash
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_kv_engine_hardening.py \
  /data/Prism-Infer/tests/test_scheduler_swap_tables.py \
  /data/Prism-Infer/tests/test_model_runner_context_reset.py \
  /data/Prism-Infer/tests/test_full_model_structure.py -s
```

结果: `17 passed in 3.57s`

```bash
PRISM_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
HF_HUB_OFFLINE=1 \
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_sequence_multimodal.py -s
```

结果: `5 passed in 4.35s`

```bash
/data/Prism-Infer/.venv-local/bin/python -m pytest -q \
  /data/Prism-Infer/tests/test_compression_off.py \
  /data/Prism-Infer/tests/test_visual_importance_scoring.py \
  /data/Prism-Infer/tests/test_visual_token_stats.py \
  /data/Prism-Infer/tests/test_analysis_schema.py -s
```

结果: `11 passed in 1.57s`

```bash
/data/Prism-Infer/.venv-local/bin/python -m compileall -q \
  /data/Prism-Infer/prism_infer \
  /data/Prism-Infer/tests \
  /data/Prism-Infer/scripts
```

结果: PASS，无编译错误输出。
