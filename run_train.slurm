#!/bin/bash
#SBATCH --job-name=medreason-lora
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"
cd "$ROOT"
python train_lora_v2.py \
  --train ft_data/train.jsonl --val ft_data/val.jsonl \
  --model MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
  --out lora_medreason --imgs-root . \
  --epochs 1 --batch 1 --grad-accum 16 --lr 1e-4 --rank 16 \
  --eval-n 200 --eval-every 400 --save-every 400
