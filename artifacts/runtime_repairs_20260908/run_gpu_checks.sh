#!/usr/bin/env bash
#SBATCH --job-name=prism-runtime-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lcpu/87120912/review-closure-20260908/runs/%x-%j.log
set -euo pipefail
STUDY_ROOT=/home/lcpu/87120912/review-closure-20260908
export PATH=/home/lcpu/87120912/venvs/prism/bin:$PATH
export PYTHONPATH=$STUDY_ROOT/source
export TRITON_CACHE_DIR=/home/lcpu/87120912/vision-dp/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/lcpu/87120912/vision-dp/cache/inductor
export CUDA_CACHE_PATH=/home/lcpu/87120912/vision-dp/cache/cuda
export HF_HOME=/home/lcpu/87120912/hf_cache
export HF_HUB_OFFLINE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export REPAIR_CAPACITY_OUTPUT=$STUDY_ROOT/runs/tp-capacity-$SLURM_JOB_ID.json
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python "$STUDY_ROOT/check_requests.py" --model /home/lcpu/87120912/models/Qwen3-VL-8B-Instruct \
    --output "$STUDY_ROOT/runs/requests-$SLURM_JOB_ID.json"
torchrun --standalone --nproc_per_node=2 "$STUDY_ROOT/check_tp_capacity.py"
