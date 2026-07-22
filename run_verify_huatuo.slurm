#!/bin/bash
#SBATCH --job-name=verify-huatuo
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"
cd "$ROOT"
python verify_lora.py \
  --val ft_data_v3/val.jsonl \
  --model models/HuatuoGPT-Vision-7B \
  --adapter lora_huatuo --imgs-root . \
  --mcq-n 400 --open-n 15
