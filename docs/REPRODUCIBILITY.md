# Prism-Infer 复现实验手册

> 更新日期：2026-07-17  
> 目标：让安装、correctness、quality 和 performance 证据分别可复现，禁止用低层
> smoke 替代高层门禁。

## 1. 证据层级

| 层级 | 需要模型 | 需要 GPU | 能证明什么 | 不能证明什么 |
|---|:---:|:---:|---|---|
| A. package smoke | 否 | 否 | 包可构建、核心依赖与 API 可导入 | 模型或 CUDA correctness |
| B. CPU/focused tests | 否 | 否或小 CUDA | schema、调度、压缩合同、分析逻辑 | full 8B logits/E2E |
| C. minimal 8B demo | 是 | 是 | 当前 snapshot 能完成一次 VL greedy | HF 对齐、稳定质量、性能 |
| D. full correctness | 是 | 是 | 模块、logits/PPL、greedy、mixed/Graph 回归 | 正式性能 |
| E. quality pair | 是/固定数据 | 是 | dense 与 compression 的相对质量和物理 KV | 通用 accuracy |
| F. formal benchmark | 是 | 独占 GPU | 指定环境/workload 的 latency、memory、throughput | 其他硬件或 online server |

任一层失败都不能用更低层 PASS 覆盖。正式记录必须包含 commit、dirty state、模型
config hash、GPU UUID、输入、sampling、backend、warmup/repeat 和同步边界。

## 2. Fresh-environment 安装

先安装与本机 CUDA 匹配的 PyTorch，再执行：

```bash
git clone https://github.com/xsmccc/Prism-Infer.git
cd Prism-Infer

python3.12 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

项目不再把 Triton/FlashAttention 作为任意版本的强制 pip 依赖：Triton 版本应由
PyTorch CUDA stack 约束，FlashAttention 应按 GPU/CUDA 平台单独安装。没有这些
backend 时，部分 CPU/SDPA correctness 仍可运行，但不能复现正式 CUDA 性能。

安装检查：

```bash
python scripts/check_environment.py
```

2026-07-17 的隔离 venv smoke（复用宿主 CUDA/PyTorch stack）真实输出格式：

```text
Prism-Infer environment check
status: PASS
runtime: Python 3.12.3, Prism 0.3.0, Torch 2.6.0a0+ecf3bae40a.nv25.1, Transformers 5.13.0
optional backends: triton=yes, flash_attn=yes
cuda: available=True, version=12.8, devices=1
gpu[0]: NVIDIA GeForce RTX 5090, cc=12.0, free/total=<dynamic> GiB
model: NOT_CHECKED
WARNING: model snapshot not checked; pass --model or set PRISM_MODEL_PATH
```

editable wheel metadata和 API import 同轮通过：

```text
Successfully built prism-infer
Successfully installed prism-infer-0.3.0
prism-infer=0.3.0
llm=prism_infer.llm.LLM
sampling=SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=False)
```

该 audit 使用 `--system-site-packages` 复用 NGC CUDA stack；宿主已有的
`nvidia-dali-cuda120`/`six` 冲突会让全环境 `pip check` 报错，但不来自 Prism
依赖。真正从零安装仍必须由用户先选择硬件匹配的 PyTorch wheel/container。

## 3. 不加载模型的可复制 smoke

```bash
python -m compileall prism_infer tests benchmarks scripts
python -m pytest -q \
  tests/test_check_environment.py \
  tests/test_analysis_schema.py \
  tests/test_visual_token_stats.py \
  tests/test_visual_importance_scoring.py \
  tests/test_compression_off.py \
  tests/test_engine_contracts.py
```

commit `d547385` 后的同组实测：

```text
Running 40 items in this shard
........................................ [100%]
40 passed in 5.11s
```

这个结果只覆盖环境检查、schema/analysis、compression-off 和 engine contracts。

## 4. 模型与 GPU preflight

```bash
export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct/snapshot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/check_environment.py \
  --model "$PRISM_MODEL_PATH" \
  --require-cuda \
  --min-free-gib 18
```

通过的 model 部分示例：

```text
model: PASS, type=qwen3_vl, weights=4 files/16.330 GiB
model path: /path/to/Qwen3-VL-8B-Instruct/snapshot
```

`--min-free-gib 18` 是基于当前 full-engine allocator peak `17.4–17.5 GiB` 的启动
门禁，不是通用硬件需求估算。2026-07-17 曾出现的外部隐藏负载会让脚本在加载权重前
失败，例如：

```text
status: FAIL
ERROR: GPU 0 has 14.696 GiB free; 18.000 GiB required
```

该事件随后恢复并由完整动态门禁关闭；不要因此删除fail-closed检查。不要降低门槛
绕过OOM，也不要把有外部utilization的timing记为formal evidence。

## 5. 最小 8B VL demo

```bash
python example.py
```

成功标准：进程正常退出、打印不超过 8 个 token IDs 和 decoded text、`llm.exit()`
释放模型资源。输出结构：

```text
Token IDs: [<integer>, ...]
Text: '<decoded text>'
```

历史 single-image correctness case 的 2-token样例为：

```text
single-image eager token_ids: [785, 2168]
single-image graph token_ids: [785, 2168]
```

这组 token 来自对应固定测试输入，不承诺与 `example.py` 的图片/prompt 相同。

## 6. Full-model correctness

先跑最窄的结构与 logits 门禁：

```bash
python -m pytest -q \
  tests/test_full_model_structure.py \
  tests/test_full_model.py \
  tests/test_full_model_vl.py \
  tests/test_llm_vl_generate.py -s
```

再跑多图、视频、mixed 和 Graph：

```bash
python -m pytest -q \
  tests/test_full_model_vl_multi_image.py \
  tests/test_full_model_vl_video.py \
  tests/test_llm_vl_mixed_batch_generate.py \
  tests/test_llm_vl_cuda_graph_decode.py \
  tests/test_vl_logits_distribution.py -s
```

最后才运行完整 suite：

```bash
python -m pytest -q tests -s \
  --junitxml=data/repro/full_regression.xml
```

当前 clean formal JUnit 是 commit `021d4e2` 的
`281 passed, 6 skipped in 297.622s`；JUnit字段为`287 tests / 0 failures / 0 errors /
6 skipped`。后续改动必须重新报告新JUnit，不能沿用该计数冒充当前结果。

## 7. KV trace 最小实验

生成 text/single-image/multi-image deterministic traces：

```bash
python scripts/run_kv_trace_samples.py \
  --model "$PRISM_MODEL_PATH" \
  --output-dir data/repro/kv_trace \
  --max-tokens 2 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024
```

离线汇总和 visual-token score：

```bash
python scripts/analyze_kv_trace.py \
  data/repro/kv_trace/single_image_description.jsonl \
  --summary-json data/repro/kv_trace/single_image_description.summary.json \
  --markdown data/repro/kv_trace/single_image_description.summary.md \
  --svg data/repro/kv_trace/single_image_description.summary.svg

python scripts/score_visual_tokens.py \
  data/repro/kv_trace/single_image_description.jsonl \
  --output-json data/repro/kv_trace/single_image_description.importance.json \
  --markdown data/repro/kv_trace/single_image_description.importance.md
```

PASS：trace on/off greedy token一致；JSONL 通过 schema 校验；summary 中 token span、
layer、K/V norm、attention/entropy 字段完整。trace timing 不能替代 uninstrumented
benchmark。

## 8. Dense 与 visual compaction 成对实验

### 8.1 Synthetic single-image smoke

```bash
python benchmarks/bench_system.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case single_image_448 \
  --modes off_graph,visual_compact_graph \
  --max-tokens 32 --warmup 2 --repeat 5 \
  --max-model-len 1024 --max-num-batched-tokens 1024 \
  --max-num-seqs 1 --num-kvcache-blocks 16 \
  --disable-prefix-caching \
  --visual-pruning-keep-ratio 0.5 \
  --visual-pruning-min-keep-tokens 32 \
  --visual-pruning-strategy attention \
  --visual-pruning-attention-last-n-layers 1 \
  --output data/repro/single_image_off_compact.jsonl
```

这条命令用于 runner smoke 和同轮 observation。单 synthetic case 不能形成质量 claim。

### 8.2 固定 7-image lexical quality gate

先下载/校验 manifest 声明的 COCO 样例：

```bash
bash scripts/download_p6_real_samples.sh
```

分别运行两个 batch：

```bash
python benchmarks/bench_system.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p6_real_samples.json \
  --case coco_fidelity_batch_a \
  --modes off_graph,visual_compact_graph \
  --max-tokens 32 --warmup 1 --repeat 1 \
  --disable-prefix-caching \
  --visual-pruning-strategy attention \
  --visual-pruning-attention-last-n-layers 1 \
  --output data/repro/coco_batch_a.jsonl

python benchmarks/bench_system.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p6_real_samples.json \
  --case coco_fidelity_batch_b \
  --modes off_graph,visual_compact_graph \
  --max-tokens 32 --warmup 1 --repeat 1 \
  --disable-prefix-caching \
  --visual-pruning-strategy attention \
  --visual-pruning-attention-last-n-layers 1 \
  --output data/repro/coco_batch_b.jsonl
```

汇总：

```bash
python scripts/summarize_p6_pruning_fidelity.py \
  data/repro/coco_batch_a.jsonl \
  data/repro/coco_batch_b.jsonl \
  --baseline-mode off_graph \
  --max-task-quality-drop 0.01 \
  --json-output data/repro/coco_quality_summary.json \
  --markdown-output data/repro/coco_quality_summary.md
```

clean `e51c16d` 的历史期望边界：

```text
token-F1 macro: 0.321635 -> 0.318347, drop 0.003288, PASS
ROUGE-L macro:  0.289116 -> 0.285406, drop 0.003710, PASS
physical prompt tokens ratio: 0.535x
active prompt bytes ratio:    0.538x
```

新环境不要求逐位复刻 latency，但 correctness/quality 超阈值时必须 FAIL，不能只报告
显存收益。

## 9. Online engine harness

```bash
python benchmarks/bench_online.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p6_internal_smoke.json \
  --case single_image_448 \
  --mode off_graph \
  --requests 8 \
  --arrival-process constant \
  --request-rate 4 \
  --max-tokens 8 \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 8 \
  --num-kvcache-blocks 16 \
  --ttft-slo-ms 1000 \
  --tpot-slo-ms 50 \
  --output data/repro/online_single_image.json
```

记录必须包含 arrival offsets、queue/TTFT/TPOT p50/p90/p99、throughput、goodput、
request terminal state、peak active/KV blocks 和 preemption。它是进程内 engine
harness，不含网络协议、序列化或多进程 server 开销。

## 10. Packed MLP P7.5 完整复现

组件correctness与formal micro：

```bash
python benchmarks/bench_packed_mlp.py \
  --batch-sizes 1,2,4,8,210,408,988 \
  --warmup 20 --repeat 100 \
  --require-formal-environment \
  --output data/p7_optimization/p75_packed_mlp_micro.json

python -m pytest -q \
  tests/test_p7_packed_mlp.py \
  tests/test_qwen3_vl.py \
  tests/test_full_model_structure.py \
  tests/test_qwen3_vl_attention_kv.py \
  tests/test_model_runner_vl_cudagraph.py \
  tests/test_model_runner_vl_prefill.py
```

full-engine单变量A/B只改变projection mode：

```bash
for projection in legacy packed; do
  python benchmarks/bench_system.py \
    --model "$PRISM_MODEL_PATH" \
    --manifest benchmarks/workloads/p6_internal_smoke.json \
    --case single_image_448 \
    --modes off_graph \
    --mlp-projection-mode "$projection" \
    --max-tokens 32 --warmup 2 --repeat 5 \
    --max-model-len 1280 --max-num-batched-tokens 2048 \
    --max-num-seqs 1 --num-kvcache-blocks 16 \
    --disable-prefix-caching \
    --output "data/p7_optimization/p75_single_${projection}.jsonl"
done
```

HF gate运行`tests/test_vl_logits_distribution.py`；online runner同样接受
`--mlp-projection-mode legacy|packed`并在schema-v2记录。Systems trace使用第8节
CUDA Profiler API做两次node capture，再由`benchmarks/analyze_nsys.py`读取SQLite。

本环境期望边界：七个micro cases bitwise exact；8个offline cells token exact且
packed/legacy TPOT为`0.9924x–0.9952x`；Systems linear `253 -> 217`、总kernels
`2,000 -> 1,964`；HF model-precision logits/PPL diff为`0`。机器可读汇总见
`data/p7_optimization/p75_summary_021d4e2.json`。这些数字不保证跨GPU复刻，也不形成
稳定E2E或online speedup claim。若baseline污染，`formal_eligible=false`记录仍不得进入
headline。

## 11. 证据保存与验收

`data/` 默认 gitignored。每次正式复现至少保存：

```text
命令与 stdout/stderr
raw JSON/JSONL 或 JUnit
机器可读 summary
commit + git_dirty
model config SHA256
GPU name/UUID/driver/CUDA
启动前 memory/utilization
warmup/repeat 与 timing scope
```

交付检查：

```bash
git diff --check
git status --short
```

只有 clean commit、完整 comparability/correctness gate 和未污染 GPU 上的结果可形成
formal performance claim。其余结果必须标成 smoke、diagnostic、dirty validation 或
blocked。
