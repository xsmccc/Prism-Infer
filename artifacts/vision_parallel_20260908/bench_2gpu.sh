#!/usr/bin/env bash
#SBATCH --job-name=prism-vision-bench
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
if [ "$#" -eq 0 ]; then
    set -- replicated data
fi
for mode in "$@"; do
    python benchmarks/bench_vision_parallel.py \
        --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
        --tensor-parallel-size 2 --vision-mode "$mode" \
        --execution-backend cuda_graph --max-model-len 4096 \
        --max-num-batched-tokens 8192 --max-num-seqs 8 --concurrent-requests 8 \
        --repeat 5 --output "$TASK_ROOT/runs/tp2-$mode-$SLURM_JOB_ID.json"
done
