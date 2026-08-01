# Reproducibility

## 1. Environment

The published results were measured with:

```text
GPU: NVIDIA GeForce RTX 5090 32 GB
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

Different GPU, driver, CUDA, PyTorch, FlashAttention, or Triton combinations
must be reported as a separate environment rather than merged into the
published results.

## 2. Installation

Install a PyTorch build that matches the host CUDA stack before installing
Prism-Infer:

```bash
git clone https://github.com/xsmccc/Prism-Infer.git
cd Prism-Infer

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[blackwell,quality,serving,dev]"
```

Set the local model snapshot:

```bash
export PRISM_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
python scripts/check_environment.py --model "$PRISM_MODEL_PATH"
```

The model directory must contain the tokenizer/processor configuration and
all weight shards. The formal runs never download a model implicitly.

## 3. Functional example

```bash
python example.py
```

The example creates a synthetic image, generates eight greedy tokens, decodes
the text, and closes the engine so GPU memory can be released.

## 4. Offline Prism cells

The two headline cases are stored in
`benchmarks/workloads/p9_headline.json`. Run each mode in a fresh process.

BF16 compiler/Graph profile:

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

Capacity profile:

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

Replace the case with `h2_video_16x448` for H2. Use
`--sample-process-memory` only for a separate memory artifact; NVML sampling
must not be enabled in the latency headline.

External adapters are:

```text
benchmarks/bench_external_vllm.py
benchmarks/bench_external_sglang.py
```

For a fair comparison, record exact prompt-token hashes, cache budget, block
size, output length, warmup/repeat, attention backend, and framework version.

## 5. Fresh-object repeat matrix

The repository includes the workload, class SLO thresholds, and runners:

```bash
P17_RUN_REVISION=repro \
  benchmarks/run_p17_repeat_matrix.sh \
  "$PRISM_MODEL_PATH" \
  data/p17_repeat_matrix

benchmarks/run_p17_vllm_repeat_matrix.sh \
  "$PRISM_MODEL_PATH" \
  data/p17_repeat_matrix
```

Each runner materializes repeat rates of 0%, 25%, 50%, 75%, and 100%, plus a
100% same-media/different-question cell. The Prism runner uses
`visual_compact_scaled_fp8_compile_graph`, SLO-aware scheduling, 220 KV pages,
and a content-addressed cache. The vLLM runner enables prefix caching and its
multimodal processor cache.

The frozen class thresholds are in
`benchmarks/configs/h3_class_slo_vllm_0251.json`. They were derived from the
vLLM 0.25.1 low-load p50 by class: 5× for TTFT and 2× for TPOT.

SGLang 0.5.15.post1 did not expose an equivalent single-node multimodal global
cache path in this environment. Its reported reference uses Radix caching and
repeated media objects; it is a useful but not feature-identical comparison.

## 6. Dual-GPU TP2

Use one host with two visible RTX 5090 GPUs. Record the interconnect separately
because direct P2P/NVLink availability can materially change collective cost:

```bash
nvidia-smi topo -m
```

Run each engine in a fresh process. For the Prism single-image output-32 cell:

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

For one text, one image, and one video request in the same batch:

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

Run `bench_external_vllm.py` and `bench_external_sglang.py` with
`--tensor-parallel-size 2` and the same case, output length, warmup/repeat,
model-length, dtype, and greedy settings. vLLM required its PyTorch-native
sampler in this Blackwell environment; record that backend rather than hiding
the startup incompatibility.

Check exact prompt-token hashes first. Prism must match vLLM token IDs and
`outputs_identical_across_repeats` before comparing TPOT. The measured SGLang
output diverges and is therefore performance-only. The Prism record must report
compile subgraph `qkv_projection_qk_norm_mrope`, KV boundary
`validated_runtime_store_and_paged_decode`, Graph capture scope
`decode_model_forward_logits_greedy`, and non-zero replay counts.

The formal online mixed path is:

```bash
python benchmarks/bench_online.py \
  --model "$PRISM_MODEL_PATH" \
  --case mixed_text_image_video \
  --mode off_graph \
  --tensor-parallel-size 2 \
  --requests 6 \
  --arrival-process burst \
  --warmup-requests 1 \
  --max-tokens 8 \
  --max-model-len 768 \
  --max-num-batched-tokens 2304 \
  --max-num-seqs 4 \
  --max-chunk-size 512 \
  --num-kvcache-blocks 16 \
  --output data/repro/tp2_online_mixed_n6.json
```

For native HTTP/SSE serving:

```bash
CUDA_VISIBLE_DEVICES=0,1 prism-serve \
  --model "$PRISM_MODEL_PATH" \
  --engine-config configs/tp2_graph.json \
  --host 127.0.0.1 \
  --port 8000
```

## 7. Correctness protocol

Before interpreting a performance cell, verify:

1. model revision, model dtype, GPU model/count/topology, driver, CUDA, and
   package versions;
2. clean source commit and complete command line;
3. exact input and prompt-token SHA256;
4. identical request order, arrivals, output length, warmup, and repeats;
5. deterministic same-shape greedy output;
6. H1/H2 isolated output hashes for optimized fast paths;
7. valid logical/physical KV lengths and page ownership counters;
8. zero terminal failure for loaded runs; and
9. process memory is released after engine exit.

The tests under `tests/` protect implementation contracts. They do not replace
the GPU benchmark protocol and are not used as a performance claim.

## 8. Evidence retention

Raw JSON, logs, Nsight traces, model weights, and dataset media are intentionally
not committed to Git because of size and licensing. Every published summary
must retain the model revision, source commit, GPU model/count and interconnect,
input hashes, and raw artifact SHA256 alongside the external evidence archive.
Physical GPU UUID is not a public comparison dimension.
