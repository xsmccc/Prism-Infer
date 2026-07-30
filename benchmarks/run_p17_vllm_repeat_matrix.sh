#!/usr/bin/env bash
#
# Run the P17 fresh-object cache matrix with vLLM's official caches enabled.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_PATH [OUTPUT_DIR]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="$1"
output_dir="${2:-${repo_root}/data/p17_repeat_matrix}"
python_bin="${repo_root}/.venv/bin/python"
mkdir -p "${output_dir}"

common_args=(
  "${repo_root}/benchmarks/bench_online_vllm.py"
  --model "${model_path}"
  --manifest "${repo_root}/benchmarks/workloads/p9_headline.json"
  --h3-profile conditional_video
  --requests 60
  --arrival-process poisson
  --request-rate 4
  --seed 20260717
  --warmup-requests 10
  --max-tokens 64
  --max-model-len 4096
  --max-num-batched-tokens 16384
  --max-num-seqs 8
  --kv-cache-memory-bytes 4294967296
  --block-size 256
  --attention-backend FLASH_ATTN
  --enable-prefix-caching
  --mm-processor-cache-gb 1
  --class-slo-file
  "${repo_root}/data/p12_online/formal/p12_class_slo_vllm_r1_ce72f63.json"
)

run_cell() {
  local label="$1"
  shift
  VLLM_USE_FLASHINFER_SAMPLER=0 "${python_bin}" "${common_args[@]}" "$@" \
    --output "${output_dir}/${label}.json" \
    >"${output_dir}/${label}.log" 2>&1
}

for rate in 0 25 50 75 100; do
  run_cell \
    "vllm_repeat${rate}_n60_s20260717_r3" \
    --media-repeat-rate "$(awk "BEGIN { print ${rate} / 100 }")"
done

run_cell \
  "vllm_repeat100_different_questions_n60_s20260717_r3" \
  --media-repeat-rate 1 \
  --vary-media-questions
