#!/usr/bin/env bash
#SBATCH --job-name=prism-vision-requests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lcpu/87120912/vision-dp/runs/%x-%j.log

set -euo pipefail
TASK_ROOT=/home/lcpu/87120912/vision-dp
SOURCE_ROOT=/home/lcpu/87120912/Prism-Infer-vision-dp
export PATH=/home/lcpu/87120912/venvs/prism/bin:$PATH
export PYTHONPATH=$SOURCE_ROOT
export TMPDIR=$TASK_ROOT/tmp
export TRITON_CACHE_DIR=$TASK_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$TASK_ROOT/cache/inductor
export CUDA_CACHE_PATH=$TASK_ROOT/cache/cuda
export HF_HOME=/home/lcpu/87120912/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
cd "$SOURCE_ROOT"
for mode in replicated data; do
    python benchmarks/check_vision_parallel_requests.py \
        --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
        --materialized-root /home/lcpu/87120912/Prism-Infer/data/quality/materialized \
        --vision-mode "$mode" --limit 20 \
        --output "$TASK_ROOT/runs/muir-$mode-$SLURM_JOB_ID.json"
done
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node --output="$TASK_ROOT/runs/vision-dp-$SLURM_JOB_ID" \
    python benchmarks/profile_vision_parallel.py \
    --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --output "$TASK_ROOT/runs/trace-requests-$SLURM_JOB_ID.json"
