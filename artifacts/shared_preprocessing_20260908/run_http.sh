#!/usr/bin/env bash
#SBATCH --job-name=prism-http-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lcpu/87120912/preprocessing/runs/%x-%j.log

set -euo pipefail
STUDY_ROOT=/home/lcpu/87120912/preprocessing
SOURCE_ROOT=$STUDY_ROOT/$1
LABEL=$1
CONFIG=${2:-engine.json}
export PATH=/home/lcpu/87120912/venvs/prism/bin:$PATH
export PYTHONPATH=$SOURCE_ROOT
export TMPDIR=$STUDY_ROOT/tmp
export TRITON_CACHE_DIR=/home/lcpu/87120912/vision-dp/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/lcpu/87120912/vision-dp/cache/inductor
export CUDA_CACHE_PATH=/home/lcpu/87120912/vision-dp/cache/cuda
export HF_HOME=/home/lcpu/87120912/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
PORT=$((18000 + SLURM_JOB_ID % 10000))
SERVER_PID=
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" || true
    fi
}
trap cleanup EXIT
python "$STUDY_ROOT/serve_preprocessing_study.py" \
    --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --config "$STUDY_ROOT/$CONFIG" --port "$PORT" \
    --output "$STUDY_ROOT/runs/server-$LABEL-$SLURM_JOB_ID.json" &
SERVER_PID=$!
# Readiness checks wait for model initialization, not a fixed warmup delay.
for attempt in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        wait "$SERVER_PID"
        exit 1
    fi
    sleep 1
done
python "$STUDY_ROOT/bench_serving_preprocessing.py" \
    --base-url "http://127.0.0.1:$PORT" --label "$LABEL" \
    --output "$STUDY_ROOT/runs/http-$LABEL-$SLURM_JOB_ID.json"
