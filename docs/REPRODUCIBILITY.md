# 运行与复现

## 环境

项目结果使用以下环境：

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
```

不同 PyTorch、CUDA、Attention Backend 和 GPU 会影响结果，建议在输出 JSON 中保留
完整版本信息。

## 安装

先安装与 CUDA 匹配的 PyTorch，再安装 Prism-Infer：

```bash
git clone https://github.com/xsmccc/Prism-Infer.git
cd Prism-Infer

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[blackwell,quality,serving,dev]"

export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
python scripts/check_environment.py --model "$PRISM_MODEL_PATH"
python example.py
```

## TP1 图片和视频测试

H1/H2 输入定义在 `benchmarks/workloads/p9_headline.json`。

```bash
python benchmarks/bench_external_prism.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p9_headline.json \
  --case h1_eight_image_448 \
  --execution-backend compile_graph \
  --compression-mode off \
  --num-kvcache-blocks 113 \
  --kvcache-block-size 256 \
  --warmup 2 \
  --repeat 5 \
  --max-tokens 128 \
  --output data/repro/prism_h1_bf16.json
```

将 case 改为 `h2_video_16x448` 可以运行视频测试。Scaled-FP8 配置：

```bash
python benchmarks/bench_external_prism.py \
  --model "$PRISM_MODEL_PATH" \
  --manifest benchmarks/workloads/p9_headline.json \
  --case h1_eight_image_448 \
  --execution-backend compile_graph \
  --compression-mode scaled_fp8_kv \
  --num-kvcache-blocks 220 \
  --kvcache-block-size 256 \
  --warmup 2 \
  --repeat 5 \
  --max-tokens 128 \
  --output data/repro/prism_h1_scaled_fp8.json
```

vLLM 和 SGLang 使用对应 adapter：

```text
benchmarks/bench_external_vllm.py
benchmarks/bench_external_sglang.py
```

比较时保持模型、prompt tokens、KV 预算、output length、warmup 和 repeat 相同。

## 双卡 TP2

先确认两张 GPU 可见：

```bash
nvidia-smi topo -m
```

单图 batch1：

```bash
python benchmarks/bench_system.py \
  --model "$PRISM_MODEL_PATH" \
  --case single_image_448 \
  --modes tp2_compile_graph \
  --tensor-parallel-size 2 \
  --max-tokens 32 \
  --warmup 1 \
  --repeat 3 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --num-kvcache-blocks 8 \
  --output data/repro/tp2_single_image_32.jsonl
```

text + image + video 混合 batch3：

```bash
python benchmarks/bench_system.py \
  --model "$PRISM_MODEL_PATH" \
  --case mixed_text_image_video \
  --modes tp2_compile_graph \
  --tensor-parallel-size 2 \
  --max-tokens 8 \
  --warmup 1 \
  --repeat 2 \
  --max-model-len 768 \
  --max-num-batched-tokens 2304 \
  --max-num-seqs 3 \
  --num-kvcache-blocks 16 \
  --output data/repro/tp2_mixed_b3.jsonl
```

外部引擎 adapter 同样支持 `--tensor-parallel-size 2`。RTX 5090 环境中，vLLM 使用
PyTorch native sampler，SGLang 使用 Triton Attention。

## 在线重复媒体测试

```bash
P17_RUN_REVISION=repro \
  benchmarks/run_p17_repeat_matrix.sh \
  "$PRISM_MODEL_PATH" \
  data/p17_repeat_matrix

benchmarks/run_p17_vllm_repeat_matrix.sh \
  "$PRISM_MODEL_PATH" \
  data/p17_repeat_matrix
```

脚本依次运行 0%、25%、50%、75%、100% 媒体重复率，以及相同媒体、更换问题的测试。

## HTTP/SSE Serving

```bash
prism-serve \
  --model "$PRISM_MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000
```

双卡配置：

```bash
CUDA_VISIBLE_DEVICES=0,1 prism-serve \
  --model "$PRISM_MODEL_PATH" \
  --engine-config configs/tp2_graph.json \
  --host 127.0.0.1 \
  --port 8000
```

## JSON 输出

Benchmark JSON 包含：

- 模型、GPU、PyTorch 和执行后端；
- TTFT、TPOT、E2E 和吞吐；
- CUDA Graph capture/replay 次数；
- torch.compile region 和首次编译时间；
- KV Cache 配置与显存；
- 输入 token、输出 token 和 SHA256。

性能测试和 Nsight Profiling 分开运行，避免 profiler 改变延迟。显存采样也使用单独
进程，不与 TPOT 测试混在一起。
