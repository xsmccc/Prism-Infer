#!/usr/bin/env bash
#SBATCH --job-name=prism-prefix-pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lcpu/87120912/preprocessing/runs/%x-%j.log

# Both servers run sequentially inside one allocation. The second pair reverses
# execution order; each gets its own process/cache and raw request file.
set -euo pipefail
STUDY_ROOT=/home/lcpu/87120912/preprocessing
export PATH=/home/lcpu/87120912/venvs/prism/bin:$PATH
export TMPDIR=$STUDY_ROOT/tmp
export TRITON_CACHE_DIR=/home/lcpu/87120912/vision-dp/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/lcpu/87120912/vision-dp/cache/inductor
export CUDA_CACHE_PATH=/home/lcpu/87120912/vision-dp/cache/cuda
export HF_HOME=/home/lcpu/87120912/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
PORT=$((18000 + SLURM_JOB_ID % 10000))
SERVER_PID=
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" || true
        SERVER_PID=
    fi
}
trap cleanup EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for CELL in baseline-1 candidate-1 candidate-2 baseline-2; do
    if [[ $CELL == baseline-* ]]; then
        export PYTHONPATH=$STUDY_ROOT/cooperative
    else
        export PYTHONPATH=$STUDY_ROOT/prefill-fused
    fi
    python "$STUDY_ROOT/serve_preprocessing_study.py" \
        --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
        --config "$STUDY_ROOT/engine.json" --port "$PORT" \
        --output "$STUDY_ROOT/runs/server-fusion-$CELL-$SLURM_JOB_ID.json" &
    SERVER_PID=$!
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
        --base-url "http://127.0.0.1:$PORT" --label "$CELL" \
        --output "$STUDY_ROOT/runs/http-fusion-$CELL-$SLURM_JOB_ID.json"
    cleanup
done
