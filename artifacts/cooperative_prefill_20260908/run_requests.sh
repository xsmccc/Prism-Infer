#!/usr/bin/env bash
#SBATCH --job-name=prism-prefill-requests
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
export PYTHONPATH=$STUDY_ROOT/cooperative
export TMPDIR=$STUDY_ROOT/tmp
export TRITON_CACHE_DIR=/home/lcpu/87120912/vision-dp/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/lcpu/87120912/vision-dp/cache/inductor
export CUDA_CACHE_PATH=/home/lcpu/87120912/vision-dp/cache/cuda
export HF_HOME=/home/lcpu/87120912/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python "$STUDY_ROOT/check_requests.py" \
    --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --config "$STUDY_ROOT/engine-cooperative.json" \
    --output "$STUDY_ROOT/runs/requests-$SLURM_JOB_ID.json"
