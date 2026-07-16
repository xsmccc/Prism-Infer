# Prism-Infer 性能问题档案

本目录保存性能调优过程中可以独立复现和审计的问题。它不是最终结果的汇总，
而是记录从“观察到异常”到“形成结论”的完整推理链，供回归、复盘和面试讲解。

## 记录原则

每个问题至少包含：

1. 现象和影响范围，避免只写最终修复。
2. 首次发现它的 workload、commit、硬件和命令。
3. 用来缩小范围的 trace、指标和对照实验。
4. 根因，以及哪些证据能排除其他解释。
5. 解决方法、为什么预期有效、实际效果。
6. 被拒绝的方法及拒绝原因。
7. correctness、质量和性能回归命令。
8. 剩余限制，防止把局部结果扩展成系统级 claim。

性能数字必须来自 warmup 后的 measured iterations。语义 profiler、Nsight capture
和调试日志用于定位，不能直接替代无 profiler 的 benchmark 数字。

## 状态

- `OPEN`：现象已确认，尚无充分根因或修复。
- `INVESTIGATING`：正在通过对照实验缩小范围。
- `RESOLVED`：修复有效且通过对应回归。
- `DOCUMENTED_LIMITATION`：不是当前范围内要修复的错误，但已固定边界。
- `REJECTED`：候选方案经证据证明不值得进入支持路径。

## 索引

- [性能调优方法](PERFORMANCE_TUNING_PLAYBOOK.md)
- [P7-000：P6.12 冻结与 claim 校正](P7-000-P6_FREEZE_AND_CLAIMS.md)
- [P7-001：跨执行协议的性能数字不能直接比较](P7-001-CROSS_PROTOCOL_COMPARISON.md)
- [P7-002：FlashInfer SM120 CUDA toolkit capability probe](P7-002-FLASHINFER_SM120_TOOLKIT_PROBE.md)
- [P7-003：CUDA Graph 后仍存在的 vLLM 差距](P7-003-GRAPH_RESIDUAL_GAP.md)

## 新 issue 模板

```markdown
# P7-XXX: 标题

- 状态:
- 首次观察 commit:
- 硬件/软件:
- 影响:

## 现象
## 如何发现
## 定位过程
## 根因
## 解决方案
## 为什么有效
## 验证
## 被拒绝的方法
## 剩余限制
## 面试表达
```
