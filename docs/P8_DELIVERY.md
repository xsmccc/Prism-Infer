# P8 项目交付 Gate Review

## 1. 身份与状态

```text
阶段: P8
标题: README、技术报告、复现手册、Known Issues与投递材料
状态: PASS
日期: 2026-07-17
安装修复 commit: 568f7bb
环境诊断 commit: d547385
静态交付 commit: 9cc1bc3
动态验收 commits: 396702d / 8293851 / 021d4e2
最终治理证据 commit: 本文件所在 commit
工作树要求: clean
```

GPU恢复后，fresh editable venv完整8B demo、当前主线full regression与P7.5
full-engine/performance均已完成。P8静态和动态出口现在同时PASS；TP2、hardware
counters、标准大规模质量集与网络server是明确的conditional/future scope。

## 2. 目标与需求映射

| Requirement / exit criterion | In scope | 实现或证据位置 | 结果 |
|---|:---:|---|:---:|
| README覆盖安装、模型、demo、验证、压缩 | yes | `README.md` | PASS |
| 模型/M-RoPE/DeepStack/KV/压缩/性能技术报告 | yes | `docs/TECHNICAL_REPORT.md` | PASS |
| 最小复现命令与日志样例 | yes | `docs/REPRODUCIBILITY.md` | PASS |
| Known Issues不隐藏未验证项 | yes | `docs/KNOWN_ISSUES.md` | PASS |
| 面试/投递材料绑定证据 | yes | `docs/APPLICATION_MATERIALS.md` | PASS |
| 新环境package/API可安装 | yes | `568f7bb`、`VERIFICATION.md` P8.1 | PASS |
| 无模型CPU/focused smoke | yes | `d547385`、`VERIFICATION.md` P8.2 | PASS |
| fresh环境完整8B最小demo | yes | `example.py`、`example_fresh_021d4e2.stdout.txt` | PASS |
| 当前主线full regression | yes | `final_full_regression_021d4e2.xml` | PASS |
| P7.5 projection动态闭环 | yes | `PERFORMANCE_REPORT.md` 6.12 | PASS |
| TP2 / hardware counters | no，conditional | KI-003/KI-004 | 未验证 |

## 3. 改动范围

```text
新增:
  docs/TECHNICAL_REPORT.md
  docs/REPRODUCIBILITY.md
  docs/KNOWN_ISSUES.md
  docs/APPLICATION_MATERIALS.md
  docs/P8_DELIVERY.md
  scripts/check_environment.py
  tests/test_check_environment.py
修改:
  README.md
  pyproject.toml
  prism_infer/config.py
  prism_infer/models/qwen3_vl.py
  benchmarks/bench_system.py
  benchmarks/bench_online.py
  docs/ROADMAP.md
  docs/VERIFICATION.md
  docs/CLAIMS.md
  docs/HISTORY.md
删除: 无
兼容性: 保留 prism_infer 0.3.0 API；修复旧 setuptools不能解析的license metadata
数据/schema迁移: system benchmark schema v6与online schema v2新增
                 mlp_projection_mode；validator继续兼容旧schema
```

安装依赖不再强制任意版本Triton/FlashAttention。Triton应由PyTorch CUDA stack约束；
FlashAttention按平台单独安装。核心直接依赖补齐NumPy、safetensors和tqdm，
Transformers固定为已验证`5.13.0`。

## 4. 验证环境

```text
GPU: NVIDIA GeForce RTX 5090
GPU UUID: GPU-989db6f6-3273-d1dd-b2b9-56cced4f30a4
Python: 3.12.3
Torch: 2.6.0a0+ecf3bae40a.nv25.01
CUDA: 12.8
Transformers: 5.13.0
模型: Qwen3-VL-8B-Instruct / 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
模型权重: 4 shards / 16.330 GiB
安装venv: /tmp/prism-install-audit-local（复用宿主CUDA/PyTorch stack）
```

历史隐藏占用恢复后，连续5次采样均为`1 MiB used / 32149 MiB free / 0% utilization`；
后续formal运行之间为`1–4 MiB / 0–2%`。所有动态数字均来自clean commit和该恢复窗口，
KI-001已关闭；未来若再次污染仍须重新通过baseline gate。

## 5. Correctness 与质量

| Gate | Reference | Workload | 结果 |
|---|---|---|:---:|
| editable metadata/build | pip/setuptools | editable wheel | PASS |
| top-level API import | installed distribution | `LLM`, `SamplingParams` | PASS |
| environment model parser | temporary fixtures | valid/missing/wrong model | 3 passed |
| CPU/focused smoke | existing independent tests | 6 test files / 40 items | PASS |
| Markdown relative links | filesystem | README + `docs/*.md` | 46/46 PASS |
| code fences | parser audit | 21 Markdown files | PASS |
| full 8B HF logits/PPL | HF reference | single/multi-image/video × 32 tokens | exact PASS |
| legacy/packed E2E | same weights/commit | 8 offline cells + 2 online cells | token exact PASS |
| packed Systems count | node-level Nsight | 31 replays/mode | `253 -> 217` linear PASS |
| full regression | current main | 287 JUnit tests | 0 failure/error |
| fresh 8B demo | isolated editable venv | deterministic image/output8 | PASS |

P8 closeout纳入P7.5的新增小幅性能claim；所有数字已同步到CLAIMS，并保留不声明
稳定E2E/online speedup的边界。

## 6. 性能与资源

```text
计时 scope: unprofiled system decode-step TPOT；warmup/repeat见schema记录
新性能 claim: packed gate/up在8个记录cell改善decode TPOT 0.483%–0.762%
Systems mechanism: linear 253 -> 217；all kernels 2000 -> 1964
明确排除: 稳定E2E latency与online goodput speedup
```

文档中的历史数字保持原commit、workload和限制。安装/CPU测试耗时不是模型性能。

## 7. 执行命令与输出

```bash
python -m compileall prism_infer tests benchmarks scripts

python -m pip install --no-deps --no-build-isolation -e /data/Prism-Infer

python -m pytest -q \
  tests/test_check_environment.py \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_visual_importance_scoring.py \
  tests/test_compression_off.py \
  tests/test_engine_contracts.py

python scripts/check_environment.py
git diff --check

PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
/tmp/prism-install-audit-local/bin/python example.py

PRISM_MODEL_PATH="$PRISM_MODEL_PATH" \
python -m pytest -q \
  --junitxml=data/p8_delivery/final_full_regression_021d4e2.xml
```

```text
editable wheel: successfully built/installed prism-infer-0.3.0
focused smoke: 40 passed in 5.31s
environment check without model: PASS
local Markdown links: 46 checked / 0 missing
fresh demo: 8 token IDs + decoded text, PASS
full regression: 281 passed, 6 skipped in 297.62s
git diff --check: PASS
```

隔离venv继承宿主`nvidia-dali-cuda120`与`six`的已有pip-check冲突；Prism wheel与
import均通过。该warning已记录在复现手册，不将其删除或包装为完全独立CUDA环境。

## 8. Claim ledger

### 可以使用

- Prism package metadata可在隔离venv构建，核心API可导入。
- P8静态交付物齐全，fresh 8B demo、完整回归和本地文档链接检查通过。
- packed gate/up在限定8-cell协议中改善decode TPOT`0.483%–0.762%`。

### 必须带限制

- “fresh-environment安装通过”只覆盖复用宿主CUDA/PyTorch stack的venv。
- `40 passed`只覆盖早期6个轻量测试文件；当前full JUnit才是完整suite证据。
- fresh venv复用同一宿主CUDA/PyTorch/driver，不等于另一台机器ABI验证。
- packed收益是decode TPOT，不是稳定E2E或online speedup。

### 禁止使用

- “P8已在另一台全新机器验证CUDA ABI/性能”。
- “packed MLP显著提升E2E或online goodput”。
- 用README重写本身升级任何P6/P7 claim。

## 9. Raw evidence

```text
package build/install: 当前会话stdout，摘要固化于VERIFICATION P8.1
CPU tests: 当前会话stdout，摘要固化于VERIFICATION P8.2
fresh demo: data/p8_delivery/example_fresh_021d4e2.{stdout,stderr}.txt
full JUnit: data/p8_delivery/final_full_regression_021d4e2.xml
P7.5 summary: data/p7_optimization/p75_summary_021d4e2.{json,md}
P7.5 raw: PERFORMANCE_REPORT.md 6.12列出的gitignored路径
```

P8正式发布时仍应把关键raw records作为GitHub release artifact上传；当前`data/`
不入Git，见KI-011。

## 10. 风险、blocker 与回退

```text
已知限制: KI-003至KI-011；KI-001/KI-002已关闭
外部conditional: TP2、NCU counters、网络server、扩大质量集
回退: pyproject安装改动可按568f7bb父提交比较；文档不影响runtime
触发回退条件: 依赖解析破坏支持环境或README命令与实际CLI不一致
```

## 11. 下一步与交接

```text
已完成: P7.5 + fresh 8B demo + full regression + requirement audit
下一平台项: TP2、NCU counters、标准质量集、真实网络server
发布状态: GitHub origin/main已同步；raw evidence另行作为release artifact保存
```

## 12. Gate Review

- [x] requirement逐项映射。
- [x] correctness先于性能；本阶段未生成污染timing。
- [x] README/ROADMAP/VERIFICATION/PERFORMANCE_REPORT/CLAIMS边界一致。
- [x] Known Issues记录动态未完成项。
- [x] package、API、CPU smoke与链接已验证。
- [x] fresh 8B demo和当前full regression。
- [x] P7.5完整动态门禁。
- [x] 最终requirement audit与claim同步。
- [x] GitHub `origin/main`已同步；大体积raw release artifact仍受KI-011约束。

结论：P8静态与动态出口PASS；GitHub同步完成，本次交付闭环。
