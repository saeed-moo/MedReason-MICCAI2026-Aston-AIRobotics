#!/bin/bash
#SBATCH --job-name=mcq-ablation
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"
python "$ROOT/mcq_ablation.py" \
  --json  "$ROOT/data/train/medreason_train_selection.json" \
  --imgs  "$ROOT/data/train/imgs" \
  --model "$ROOT/MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL" \
  --n 200
