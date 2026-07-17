# P8 项目交付 Gate Review

## 1. 身份与状态

```text
阶段: P8
标题: README、技术报告、复现手册、Known Issues与投递材料
状态: PARTIAL
日期: 2026-07-17
安装修复 commit: 568f7bb
环境诊断 commit: d547385
静态交付 commit: 9cc1bc3
治理证据 commit: 本文件所在 commit
工作树要求: clean
```

`PARTIAL` 的唯一主动态原因不是文档缺失：fresh环境完整8B demo、当前主线full
regression和P7.5 full-engine/performance仍被KI-001隐藏GPU workload阻塞。CPU smoke
不能替代这些门禁。

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
| fresh环境完整8B最小demo | yes | `example.py`、KI-001 | BLOCKED |
| 当前主线full regression | yes | KI-001/P7.5 | BLOCKED |
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
  docs/ROADMAP.md
  docs/VERIFICATION.md
  docs/CLAIMS.md
  docs/HISTORY.md
删除: 无
兼容性: 保留 prism_infer 0.3.0 API；修复旧 setuptools不能解析的license metadata
数据/schema迁移: environment report schema v1，仅新增工具输出
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
安装venv: /tmp/prism-install-audit-20260717（复用宿主CUDA/PyTorch stack）
```

GPU环境不稳定：一次检查为`30.901/31.396 GiB` free，随后隐藏占用恢复；五次采样
均为`17,102 MiB used / 15,049 MiB free / 22–33% utilization`。因此本阶段没有生成
新的full-model或性能数字。

## 5. Correctness 与质量

| Gate | Reference | Workload | 结果 |
|---|---|---|:---:|
| editable metadata/build | pip/setuptools | editable wheel | PASS |
| top-level API import | installed distribution | `LLM`, `SamplingParams` | PASS |
| environment model parser | temporary fixtures | valid/missing/wrong model | 3 passed |
| CPU/focused smoke | existing independent tests | 6 test files / 40 items | PASS |
| Markdown relative links | filesystem | README + `docs/*.md` | 46/46 PASS |
| code fences | parser audit | 21 Markdown files | PASS |
| full 8B after packed MLP | HF / engine | text+VL | BLOCKED |

P8没有改变模型数学路径，也没有新增质量/performance claim；技术报告只汇总CLAIMS中
已有clean evidence。

## 6. 性能与资源

```text
计时 scope: 不适用；P8未运行正式GPU benchmark
新性能 claim: 无
GPU blocker: KI-001
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
```

```text
editable wheel: successfully built/installed prism-infer-0.3.0
focused smoke: 40 passed in 5.31s
environment check without model: PASS
local Markdown links: 46 checked / 0 missing
git diff --check: PASS
```

隔离venv继承宿主`nvidia-dali-cuda120`与`six`的已有pip-check冲突；Prism wheel与
import均通过。该warning已记录在复现手册，不将其删除或包装为完全独立CUDA环境。

## 8. Claim ledger

### 可以使用

- Prism package metadata可在隔离venv构建，核心API可导入。
- P8静态交付物齐全，CPU/focused smoke和本地文档链接检查通过。

### 必须带限制

- “fresh-environment安装通过”只覆盖复用宿主CUDA/PyTorch stack的venv。
- `40 passed`只覆盖6个轻量测试文件，不是full suite。
- 技术报告性能数字是历史clean evidence，不是2026-07-17受污染GPU上的新测量。

### 禁止使用

- “P8已在全新机器完整运行8B”。
- “packed MLP已经提升TPOT”。
- 用README重写本身升级任何P6/P7 claim。

## 9. Raw evidence

```text
package build/install: 当前会话stdout，摘要固化于VERIFICATION P8.1
CPU tests: 当前会话stdout，摘要固化于VERIFICATION P8.2
environment JSON: /tmp/prism-environment-report-20260717.json（临时）
历史模型/performance raw evidence: 各报告记录的gitignored data路径
```

P8正式发布时仍应把关键raw records作为GitHub release artifact上传；当前`data/`
不入Git，见KI-011。

## 10. 风险、blocker 与回退

```text
已知限制: KI-001至KI-011
外部blocker: 隐藏GPU allocation/utilization
回退: pyproject安装改动可按568f7bb父提交比较；文档不影响runtime
触发回退条件: 依赖解析破坏支持环境或README命令与实际CLI不一致
```

## 11. 下一步与交接

```text
最高优先级: 稳定独占GPU后恢复P7.5
恢复前置: scripts/check_environment.py --require-cuda --min-free-gib 18多次PASS
恢复内容: paired micro + full HF/E2E/online + TPOT + Systems trace
随后: fresh 8B demo、当前主线full regression、requirement-by-requirement audit
最终: 更新所有claim/状态，推送GitHub并确认local/remote同步
```

## 12. Gate Review

- [x] requirement逐项映射。
- [x] correctness先于性能；本阶段未生成污染timing。
- [x] README/ROADMAP/VERIFICATION/PERFORMANCE_REPORT/CLAIMS边界一致。
- [x] Known Issues记录动态未完成项。
- [x] package、API、CPU smoke与链接已验证。
- [ ] fresh 8B demo和当前full regression。
- [ ] P7.5完整动态门禁。
- [ ] 最终发布artifact和GitHub同步。

结论：P8静态交付PASS，P8阶段整体`PARTIAL`。
