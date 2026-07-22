#!/bin/bash
#SBATCH --job-name=both-checks
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"
cd "$ROOT"
echo "########## HYBRID v2 (exact prompts) ##########"
python validate_hybrid_v2.py \
  --val ft_data_v3/val.jsonl \
  --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
  --adapter lora_medreason_v4 --mcq-n 400 --open-n 202
echo "########## MCQ SELF-CONSISTENCY ##########"
python validate_mcq_selfconsistency.py \
  --val ft_data_v3/val.jsonl \
  --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
  --adapter lora_medreason_v4 --n 400 --k 5 --temp 0.7
