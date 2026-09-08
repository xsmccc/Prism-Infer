# 运行与复现

## 1. 环境

以下为历史重复视觉上下文工作集的环境；本轮运行时修复记录见
[RUNTIME_FIXES_20260907.md](RUNTIME_FIXES_20260907.md)：

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

### 当前 TP1 服务配置

Qwen3-VL-8B 的单卡 FP8 服务可直接使用仓库中的配置文件：

```bash
prism-serve --model "$PRISM_MODEL_PATH" \
  --engine-config configs/tp1_fp8.json \
  --host 127.0.0.1 --port 8000
```

[`configs/tp1_fp8.json`](../configs/tp1_fp8.json) 使用 Scaled-FP8 KV、Decode CUDA Graph
和 Prefix Cache，`max_model_len=4096`、`max_num_batched_tokens=4096`、
`max_num_seqs=4`，`image_max_pixels=200704`，即 448×448 的像素上限。
KV 页数按可用显存自动确定，不代表历史约 4 GiB 或 220-page 测量配置；复现某张结果表时
应使用该实验保存的配置。此服务配置不开启视觉 Token Pruning、Vision 输出缓存、
cooperative Prefill、FlashInfer 或 `torch.compile`。

`enable_chunked_prefill` 现在默认 `false`。显式开启时，当前策略仍把一条请求中从首个到
最后一个视觉 token 的区间整体处理；`max_chunk_size` 小于该区间会拒绝请求，不会自动
拆成单张图片。Prefix 部分命中后只计算未覆盖图片，是缓存复用路径，不是新的 token
分块算法。cooperative Prefill 是另一选项，也默认关闭，其历史延迟取舍见
[交错执行记录](PREFILL_INTERLEAVING.md)。

仅选择 `cuda_graph` 不启用编译。需要使用 TP1 历史编译路径时，需同时配置
`execution_backend=compile_graph`、`decode_compile_region=stateless` 和
`logits_precision=selective_fp32`。该路径编译 batch-1 O-proj 与 FP8 LM-head 候选投影，
保留原始 LM-head 权重用于 FP32 重排，不是把整个 Decoder 交给 `torch.compile`。
重排只覆盖 Top-64 候选，没有低 margin 全词表回退；配置和数值含义见
[Architecture](ARCHITECTURE.md#3-torchcompile-与-cuda-graph)。

`enable_flashinfer_paged`、`enable_flashinfer_decode` 仅用于显式选择 FlashInfer 的
未量化 BF16/FP16 KV 路径。后端不可用、与 FP8 KV 不兼容，或将 FlashInfer Decode 与
`attention` compile region 组合时会报错，不会把请求静默切换到其他后端。

## 2. 查看仓库内证据

模型权重与数据集媒体不随仓库分发。先区分数据来源，再查看请求记录：

- `artifacts/cache_pressure_20260813/`：int64 压实修复后的历史配对实验与 MuirBench 质量记录。
- `artifacts/review_fixes_20260907/`：本轮 Scaled-FP8 Prefill 执行路径观察。
- `artifacts/working_set/`：早期测量原件；Compact 延迟和剪枝质量含已知压实缺陷，
  不用于当前算法取舍或排名。

例如在仓库根目录展开修复后的逐题记录：

```bash
gzip -dk artifacts/cache_pressure_20260813/muir_dense_media_first.json.gz
gzip -dk artifacts/cache_pressure_20260813/muir_uniform_reuse.json.gz
```

逐题 `sample_id` 用于配对，`score.strict_score` 是严格答案分数，
`quality_dropped_visual_tokens > 0` 标识实际删除样本。比较 49 题时，Dense 也必须取同一批
Sample ID，不能用它自身的删除计数筛选。统计表集中在 [Results](RESULTS.md)。

`artifacts/working_set/highlights.json` 已更正为带来源日期的摘要。旧导出中的
`SHA256SUMS` 仅记录当时的文件状态，不是修改文档后必须重新生成的运行条件。

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

历史 Plan 与 Dense Page 清单位于 `artifacts/working_set/protocol/`。

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

可选 stage 包括 `muir_dense_official`、`muir_dense_media_first`、
`muir_attention_per_question`、`muir_attention_first_reuse`、`muir_uniform_reuse`，以及
DocVQA/MVBench 的 Dense 与 Uniform 对照。按需要选择，不要求每次修改重跑全部配置。

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
