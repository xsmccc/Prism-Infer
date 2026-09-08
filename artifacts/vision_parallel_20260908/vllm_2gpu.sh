#!/usr/bin/env bash
#SBATCH --job-name=prism-vllm-parallel
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
export PATH=/home/lcpu/87120912/venvs/vllm/bin:$PATH
export PYTHONPATH=$SOURCE_ROOT
export TMPDIR=$TASK_ROOT/tmp
export TRITON_CACHE_DIR=$TASK_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$TASK_ROOT/cache/inductor
export CUDA_CACHE_PATH=$TASK_ROOT/cache/cuda
export VLLM_CACHE_ROOT=$TASK_ROOT/cache/vllm
export HF_HOME=/home/lcpu/87120912/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
cd "$SOURCE_ROOT"
python benchmarks/bench_vllm_parallel.py \
    --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --tp 2 --pp 1 --encoder-mode data --max-num-seqs 8 --concurrent-requests 8 \
    --repeat 5 --output "$TASK_ROOT/runs/vllm-tp2-$SLURM_JOB_ID.json"
python benchmarks/bench_vllm_parallel.py \
    --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --tp 1 --pp 2 --encoder-mode weights --max-num-seqs 8 --concurrent-requests 8 \
    --repeat 5 --output "$TASK_ROOT/runs/vllm-pp2-$SLURM_JOB_ID.json"
