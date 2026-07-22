#!/bin/bash
#SBATCH --job-name=medreason-qtest
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"

python "$ROOT/quality_test.py" \
  --json  "$ROOT/data/train/medreason_train_selection.json" \
  --imgs  "$ROOT/data/train/imgs" \
  --model "$ROOT/MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL" \
  --mcq-n 300 --open-n 40
