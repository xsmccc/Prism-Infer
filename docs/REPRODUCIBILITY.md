# 运行与复现

## 1. 环境

重复视觉上下文工作集使用：

```text
GPU: RTX 5090 32 GB
Driver: 580.105.08
CUDA: 13.0
Python: 3.12.3
PyTorch: 2.11.0+cu130
Transformers: 5.14.1
vLLM: 0.25.1
SGLang: 0.5.15.post1
Model: Qwen3-VL-8B-Instruct
Model revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

Prism、vLLM 和 SGLang 使用独立 Python 环境，避免 Torch、Transformers 与 Attention
backend 依赖互相覆盖。请求级 JSON 保存实际软件版本、GPU、模型配置哈希、Prompt 身份和
运行参数。

```bash
git clone https://github.com/xsmccc/Prism-Infer.git
cd Prism-Infer

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[blackwell,quality,serving,dev]"

export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
python scripts/check_environment.py --model "$PRISM_MODEL_PATH"
```

## 2. 检查仓库内证据

仓库包含共用 Plan、请求级性能与质量记录、派生表、主图和 Prefix-hit Trace。模型权重与
数据集媒体不随仓库分发。

```bash
cd artifacts/working_set
sha256sum -c SHA256SUMS
```

压缩记录可单独展开：

```bash
gzip -dk performance/performance_raw/prism_compact_prefix_pressure.json.gz
gzip -dk quality/quality_raw/muir_uniform_reuse.json.gz
```

质量请求记录中的 `materialization_verification.selection_sha256` 指向测量当时的选择文件。
原件保存在 `protocol/quality_raw/quality_selection.json.gz`；解压后 SHA256 为
`c511ab44dca420a5b4ef65ae378104ce046f21f81703203a26a73620f9a9651e`。当前
`benchmarks/workloads/` 下的文件使用清理后的命名和元数据，用于新的运行，不替代历史
测量身份。

适合快速检查的文件：

- `highlights.json`：主结果、质量代价和适用范围；
- `performance/working_set_summary.json`：完整工作集机器可读汇总；
- `performance/working_set_prism_ablation.csv`：Prism 内部对照；
- `performance/working_set_engine_comparison.csv`：三引擎比较；
- `trace/trace_audit.json`：冷请求与 Prefix 命中路径观察。

## 3. 数据准备

MuirBench 与 DocVQA 按固定公开 Sample ID 物化：

```bash
python scripts/materialize_quality_data.py \
  --raw-root data/quality/raw \
  --output-root data/quality/materialized \
  --selection-output benchmarks/workloads/quality_selection.json

python scripts/verify_quality_data.py \
  --raw-root data/quality/raw \
  --materialized-root data/quality/materialized
```

MVBench 先按精确媒体身份建立同视频多问题子集，再通过 HTTP Range、CRC 和 SHA256 物化
选中的 archive members：

```bash
python scripts/build_mvbench_repeated_subset.py \
  --raw-root data/quality/raw \
  --output-root data/quality/mvbench_repeated \
  --selection-output data/quality/mvbench_repeated/mvbench_repeated_selection.json

python scripts/materialize_mvbench_media.py \
  --output-root data/quality/mvbench_repeated \
  --selection-output data/quality/mvbench_repeated/mvbench_repeated_selection.json \
  --exclude-unavailable-manual
```

最终子集包含 123 个可验证视频和 252 个问题。仓库保存选择、archive revision、CRC 与
SHA256，但不重新分发数据集媒体。

## 4. 生成 Working-set Plan

Dense Scaled-FP8 预运行真实加载每个媒体组并原子记录 Prefix pages，随后构造 `fit`、
`knee` 和 `pressure` 请求流：

```bash
python benchmarks/build_working_set_plan.py \
  --model "$PRISM_MODEL_PATH" \
  --model-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --materialized-root data/quality/materialized \
  --page-artifact data/working_set/muirbench_dense_prefix_pages_8192.json \
  --output data/working_set/muirbench_working_set_plan.json
```

当前仓库产物身份：

```text
dense pages SHA256: 028f471748d7431f63b7cfbb859fc73680aa7a8b97b4c9ff9c4d969eff7ebbef
plan SHA256:        74102cb70c6a62cdb62c1c6ed72c92a878d3a11eda4c43da101c2305fabb2739
```

Plan 只选择至少包含两个不同问题的媒体组，并记录 `available_questions`、
`observed_questions` 与 `measured_question_switches`。当前三个工作集的 600 条测量请求都
切换问题。

## 5. 运行三引擎工作集

```bash
python benchmarks/run_working_set_matrix.py \
  --model "$PRISM_MODEL_PATH" \
  --plan data/working_set/muirbench_working_set_plan.json \
  --materialized-root data/quality/materialized \
  --output-dir data/working_set/matrix \
  --prism-python /path/to/prism/.venv/bin/python \
  --vllm-python /path/to/vllm/.venv/bin/python \
  --sglang-overlay /path/to/sglang/overlay
```

矩阵运行 Prism 的 `vision_only`、`dense_prefix`、`compact_prefix`，以及 vLLM、SGLang，
并覆盖三个工作集。中断后可对同一输出目录增加 `--resume`；程序先核对已有记录的 Plan、
模型、配置和 SHA256，再只运行缺失组合。

```bash
python benchmarks/summarize_working_set.py \
  data/working_set/matrix/raw/*.json \
  --output-dir data/working_set/summary
```

汇总器核对 KV 字节预算、模型配置、请求轨迹、问题覆盖和跨引擎 post-tokenization Prompt
哈希；无法从外部框架直接获得的缓存计数保留为 `unavailable`。

## 6. 质量对照

标准配置调用格式：

```bash
python benchmarks/bench_working_set_quality.py run \
  --model "$PRISM_MODEL_PATH" \
  --stage muir_uniform_reuse \
  --output data/working_set/quality/muir_uniform_reuse.json \
  --materialized-root data/quality/materialized \
  --raw-root data/quality/raw
```

需要运行的配置：

```text
muir_dense_official
muir_dense_media_first
muir_attention_per_question
muir_attention_first_reuse
muir_uniform_reuse
docvqa_dense
docvqa_uniform
mvbench_dense
mvbench_uniform
```

两个 Attention 配置把选择和 replay 放在独立进程，以避免同时保留两份模型状态：

```bash
python benchmarks/bench_working_set_quality.py run \
  --model "$PRISM_MODEL_PATH" \
  --stage muir_attention_per_question \
  --phase selection \
  --output data/working_set/quality/muir_attention_per_question.json

python benchmarks/bench_working_set_quality.py run \
  --model "$PRISM_MODEL_PATH" \
  --stage muir_attention_per_question \
  --phase replay --resume \
  --output data/working_set/quality/muir_attention_per_question.json
```

MVBench 配置将 `--materialized-root` 指向 `data/quality/mvbench_repeated`，并增加：

```bash
--selection data/quality/mvbench_repeated/mvbench_repeated_selection.json
```

汇总：

```bash
python benchmarks/bench_working_set_quality.py summarize \
  --input data/working_set/quality/*.json \
  --output data/working_set/quality_summary.json
```

压实质量单独汇总真正删除过 token 的配对样本，避免未压实样本稀释结果。

## 7. Prefix-hit Nsight Trace

```bash
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output=data/working_set/trace/prefix_hit \
  python benchmarks/trace_working_set_prefix.py \
    --model "$PRISM_MODEL_PATH" \
    --working-set-plan data/working_set/muirbench_working_set_plan.json \
    --materialized-root data/quality/materialized \
    --output data/working_set/trace/trace_evidence.json

nsys export \
  --type=sqlite \
  --output=data/working_set/trace/prefix_hit.sqlite \
  data/working_set/trace/prefix_hit.nsys-rep
```

```bash
python benchmarks/analyze_nsys.py \
  data/working_set/trace/prefix_hit.sqlite \
  --engine-range prism::prefix_hit_request \
  --prefill-steps 0 \
  --target-range prism::cold_request \
  --target-range prism::prefix_hit_request \
  --output data/working_set/trace/nsys_summary.json \
  --quiet

python benchmarks/audit_working_set_prefix_trace.py \
  --sqlite data/working_set/trace/prefix_hit.sqlite \
  --evidence data/working_set/trace/trace_evidence.json \
  --output data/working_set/trace/trace_audit.json
```

Trace 用于确认 Prefix 命中请求没有 Vision/DeepStack、确实跳过 hydration、复用了公共
tokens 且没有 stale fallback；它带有 profiler 开销，不与在线 TTFT 表混用。

## 8. Decode 与 TP2 补充测量

TP1 图片/视频输入定义在 `benchmarks/workloads/decode_cases.json`。Prism 入口示例：

```bash
python benchmarks/bench_external_prism.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/decode_cases.json \
  --case eight_image_448 \
  --max-tokens 128 \
  --warmup 2 \
  --repeat 5 \
  --execution-backend compile_graph \
  --output data/decode/prism_eight_image_448.json
```

vLLM 与 SGLang 使用 `bench_external_vllm.py`、`bench_external_sglang.py` 消费同一
manifest。TP2 使用 `benchmarks/bench_system.py --tensor-parallel-size 2`。性能测量与
Nsight capture 分开运行；Process peak 使用 NVML 进程显存采样。
