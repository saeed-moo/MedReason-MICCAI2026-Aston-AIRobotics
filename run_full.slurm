#!/bin/bash
#SBATCH --job-name=medreason-full
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"

python "$ROOT/medreason_infer.py" \
  --json  "$ROOT/data/validation/medreason_validation_participant_facing.json" \
  --imgs  "$ROOT/data/validation/imgs" \
  --model "$ROOT/MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL" \
  --out   "$ROOT/outputs/results_full.json"
