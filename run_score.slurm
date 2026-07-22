#!/bin/bash
#SBATCH --job-name=score-open
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
cd "$ROOT"
echo "===== QWEN v4 ====="
python score_open.py --val ft_data_v3/val.jsonl \
  --model MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
  --adapter lora_medreason_v4 --imgs-root . --n 202
echo "===== HUATUOGPT ====="
python score_open.py --val ft_data_v3/val.jsonl \
  --model models/HuatuoGPT-Vision-7B \
  --adapter lora_huatuo --imgs-root . --n 202
