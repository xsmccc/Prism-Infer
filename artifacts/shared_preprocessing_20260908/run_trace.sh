#!/usr/bin/env bash
#SBATCH --job-name=prism-preprocess-trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lcpu/87120912/preprocessing/runs/%x-%j.log

set -euo pipefail
STUDY_ROOT=/home/lcpu/87120912/preprocessing
export PATH=/home/lcpu/87120912/venvs/prism/bin:$PATH
export PYTHONPATH=$STUDY_ROOT/candidate
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
PID_FILE=$STUDY_ROOT/runs/profile-server-$SLURM_JOB_ID.pid
SERVER_PID=
PROFILE_PID=
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$PROFILE_PID" ]; then
        wait "$PROFILE_PID" || true
    fi
    rm -f -- "$PID_FILE"
}
trap cleanup EXIT
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
    --output="$STUDY_ROOT/runs/preprocessing-$SLURM_JOB_ID" \
    python "$STUDY_ROOT/serve_preprocessing_study.py" \
    --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --config "$STUDY_ROOT/engine.json" --port "$PORT" --nvtx --pid-file "$PID_FILE" \
    --output "$STUDY_ROOT/runs/server-trace-$SLURM_JOB_ID.json" &
PROFILE_PID=$!
for attempt in $(seq 1 180); do
    if [ -f "$PID_FILE" ]; then
        read -r SERVER_PID < "$PID_FILE" || true
    fi
    if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
        break
    fi
    if ! kill -0 "$PROFILE_PID" 2>/dev/null; then
        wait "$PROFILE_PID"
        exit 1
    fi
    sleep 1
done
python "$STUDY_ROOT/bench_serving_preprocessing.py" \
    --base-url "http://127.0.0.1:$PORT" --label trace --repeat 1 \
    --output "$STUDY_ROOT/runs/http-trace-$SLURM_JOB_ID.json"
