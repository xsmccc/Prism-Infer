# 运行与复现

## 1. 环境

主要工作集结果使用：

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

Prism、vLLM 和 SGLang 使用独立 Python 环境，避免它们对 Torch、Transformers 和
Attention backend 的依赖互相覆盖。原始 JSON 保存实际版本、模型、输入身份和配置。

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

## 2. 检查仓库内的结果

仓库包含共用 plan、15 个请求级性能记录、9 个质量记录和代表性 Nsight Trace：

```bash
cd artifacts/working_set
sha256sum -c SHA256SUMS
```

压缩 JSON 可以单独展开：

```bash
gzip -dk performance/performance_raw/prism_compact_prefix_pressure.json.gz
gzip -dk quality/quality_raw/muir_uniform_reuse.json.gz
```

`performance/working_set_summary.json` 与 `quality/quality_summary.json` 是机器可读汇总；
Markdown、CSV 和 PNG 是派生展示。

## 3. 数据准备

MuirBench 与 DocVQA 从固定 Sample IDs 物化：

```bash
python scripts/materialize_p9_quality.py \
  --raw-root /path/to/p9_quality/raw \
  --output-root /path/to/p9_quality/materialized \
  --selection-output /path/to/p9_quality/materialized/selection.json
```

MVBench 先按精确媒体身份建立同视频多问题子集，再用 HTTP Range、CRC 和 SHA256 物化
选中的 archive members：

```bash
python scripts/build_mvbench_repeated_subset.py \
  --raw-root /path/to/p9_quality/raw \
  --output-root /path/to/mvbench_repeated \
  --selection-output /path/to/mvbench_repeated/p9_mvbench_repeated_selection.json

python scripts/materialize_p9_mvbench_media.py \
  --output-root /path/to/mvbench_repeated \
  --selection-output /path/to/mvbench_repeated/p9_mvbench_repeated_selection.json \
  --exclude-unavailable-manual
```

最终选择包含 123 个可验证视频、252 个问题。仓库记录选择、archive revision、CRC 与
SHA256，但不分发数据集媒体。

## 4. 生成 Working-set Plan

Dense Scaled-FP8 预运行真实加载每个媒体组并原子保存 page 进度，随后生成三个请求流：

```bash
python benchmarks/build_working_set_plan.py \
  --model "$PRISM_MODEL_PATH" \
  --model-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --materialized-root /path/to/p9_quality/materialized \
  --page-artifact data/working_set/muirbench_dense_prefix_pages_8192.json \
  --output data/working_set/muirbench_working_set_plan.json
```

最终身份：

```text
dense pages SHA256: 028f471748d7431f63b7cfbb859fc73680aa7a8b97b4c9ff9c4d969eff7ebbef
plan SHA256:        3af9e6e29656c9fe93d08c760e64a71ee4c6deb67782cbe228cc27cd56ff984a
```

## 5. 运行性能矩阵

```bash
python benchmarks/run_working_set_matrix.py \
  --model "$PRISM_MODEL_PATH" \
  --plan data/working_set/muirbench_working_set_plan.json \
  --materialized-root /path/to/p9_quality/materialized \
  --output-dir data/working_set/matrix \
  --prism-python /path/to/prism/.venv/bin/python \
  --vllm-python /path/to/vllm/.venv/bin/python \
  --sglang-overlay /path/to/sglang/overlay
```

矩阵包含 Prism 的 `vision_only`、`dense_prefix`、`compact_prefix`，以及 vLLM、SGLang，
覆盖 `fit`、`knee`、`pressure`，共 15 个 cell。实例中断后对同一输出目录增加
`--resume`；程序会核对已完成 cell 的 plan、模型、配置和 SHA256，只运行缺失 cell。

```bash
python benchmarks/summarize_working_set.py \
  data/working_set/matrix/raw/*.json \
  --output-dir data/working_set/summary
```

完整汇总要求 15/15 cell，不设置人为性能阈值。

## 6. 质量对照

标准 stage 调用格式：

```bash
python benchmarks/bench_working_set_quality.py run \
  --model "$PRISM_MODEL_PATH" \
  --stage muir_uniform_reuse \
  --output data/working_set/quality/muir_uniform_reuse.json \
  --materialized-root /path/to/p9_quality/materialized \
  --raw-root /path/to/p9_quality/raw \
  --phase all
```

用同一格式依次运行：

```text
muir_dense_official
muir_dense_media_first
muir_uniform_reuse
docvqa_dense
docvqa_uniform
```

`muir_attention_per_question` 与 `muir_attention_first_reuse` 必须把 Attention 选择和 replay
放在独立进程；第一次使用 `--phase selection`，第二次使用 `--phase replay --resume`。

`mvbench_dense` 与 `mvbench_uniform` 使用相同命令，但把 `--materialized-root` 指向
MVBench 物化目录，并增加：

```bash
--selection /path/to/mvbench_repeated/p9_mvbench_repeated_selection.json
```

九个 stage 完成后做严格配对汇总：

```bash
python benchmarks/bench_working_set_quality.py summarize \
  --input data/working_set/quality/*.json \
  --output data/working_set/quality_summary.json
```

汇总只用候选配置真正物理删除过 token 的样本评价压实质量；没有删除的样本不会稀释
结果。DocVQA 因而明确报告 0 个 actual-deletion samples。

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
    --materialized-root /path/to/p9_quality/materialized \
    --output data/working_set/trace/trace_evidence.json

nsys export \
  --type=sqlite \
  --output=data/working_set/trace/prefix_hit.sqlite \
  data/working_set/trace/prefix_hit.nsys-rep
```

分析与语义审计：

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

审计检查 cold range 有 Vision、hit range 无 Vision/DeepStack、命中跳过 hydration、复用
公共 tokens 且没有 stale fallback。Trace 带有 profiler 开销，不与在线 TTFT 表混用。

## 8. Decode 与 TP2 补充测量

TP1 图片/视频输入定义在 `benchmarks/workloads/p9_headline.json`，入口为
`benchmarks/bench_external_prism.py` 及对应 vLLM/SGLang adapter。TP2 使用
`benchmarks/bench_system.py --tensor-parallel-size 2`。完整固定参数见原始结果 JSON；
性能测量与 Nsight capture 分开运行，Process peak 由 NVML 进程显存采样记录，不从整卡
`memory.used` 推断。
