#!/usr/bin/env bash
#
# Run the P17 content-reuse matrix on the frozen H3 conditional-video trace.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_PATH [OUTPUT_DIR]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="$1"
output_dir="${2:-${repo_root}/data/p17_repeat_matrix}"
python_bin="${repo_root}/.venv/bin/python"
revision="${P17_RUN_REVISION:-r3}"
mkdir -p "${output_dir}"

common_args=(
  "${repo_root}/benchmarks/bench_online.py"
  --model "${model_path}"
  --manifest "${repo_root}/benchmarks/workloads/p9_headline.json"
  --case guardrail_text_short
  --mode visual_compact_scaled_fp8_compile_graph
  --h3-profile conditional_video
  --requests 60
  --arrival-process poisson
  --request-rate 4
  --seed 20260717
  --warmup-requests 10
  --max-tokens 64
  --online-cpu-intraop-threads 8
  --max-model-len 4096
  --max-num-batched-tokens 4096
  --max-num-seqs 16
  --max-chunk-size 2048
  --scheduler-policy slo_aware
  --num-kvcache-blocks 220
  --kvcache-block-size 256
  --enable-prefix-caching
  --enable-visual-embedding-cache
  --enable-cooperative-prefill
  --cooperative-prefill-layer-quantum 1
  --cooperative-prefill-vision-block-quantum 1
  --visual-pruning-strategy uniform
  --visual-pruning-keep-ratio 0.6
  --visual-pruning-min-keep-tokens 768
  --visual-pruning-video-min-keep-tokens 256
  --class-slo-file
  "${repo_root}/data/p12_online/formal/p12_class_slo_vllm_r1_ce72f63.json"
)

run_cell() {
  local label="$1"
  shift
  CUDA_MODULE_LOADING=LAZY "${python_bin}" "${common_args[@]}" "$@" \
    --output "${output_dir}/${label}.json" \
    >"${output_dir}/${label}.log" 2>&1
}

for rate in 0 25 50 75 100; do
  run_cell \
    "p17_repeat${rate}_n60_s20260717_${revision}" \
    --media-repeat-rate "$(awk "BEGIN { print ${rate} / 100 }")"
done

run_cell \
  "p17_repeat100_different_questions_n60_s20260717_${revision}" \
  --media-repeat-rate 1 \
  --vary-media-questions
