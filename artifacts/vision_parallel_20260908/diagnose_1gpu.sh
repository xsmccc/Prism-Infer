#!/usr/bin/env bash
#SBATCH --job-name=prism-vision-diagnose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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
cd "$SOURCE_ROOT"
python benchmarks/diagnose_vision_partition.py \
    --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --cases unequal_images --output "$TASK_ROOT/runs/vision-partition-$SLURM_JOB_ID.json"
