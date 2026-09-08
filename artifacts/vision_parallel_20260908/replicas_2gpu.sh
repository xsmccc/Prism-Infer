#!/usr/bin/env bash
#SBATCH --job-name=prism-two-replicas
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
export PYTHONUNBUFFERED=1
cd "$SOURCE_ROOT"
python benchmarks/bench_vision_replicas.py \
    --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --output-dir "$TASK_ROOT/runs/replicas-$SLURM_JOB_ID"
