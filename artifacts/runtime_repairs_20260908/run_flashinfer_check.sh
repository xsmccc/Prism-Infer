#!/usr/bin/env bash
#SBATCH --job-name=prism-flashinfer-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lcpu/87120912/review-closure-20260908/runs/%x-%j.log
set -euo pipefail
STUDY_ROOT=/home/lcpu/87120912/review-closure-20260908
export PATH=/home/lcpu/87120912/venvs/prism/bin:$PATH
export PYTHONPATH=$STUDY_ROOT/source
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export FLASHINFER_WORKSPACE_DIR=$STUDY_ROOT/flashinfer
python "$STUDY_ROOT/check_flashinfer.py" --output "$STUDY_ROOT/runs/flashinfer-$SLURM_JOB_ID.json"
