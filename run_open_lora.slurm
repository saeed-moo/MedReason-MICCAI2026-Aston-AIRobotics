#!/bin/bash
#SBATCH --job-name=open-lora
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"
cd "$ROOT"
python train_lora_v3.py \
  --train ft_data_open/train.jsonl --val ft_data_open/val.jsonl \
  --model MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
  --out lora_open --imgs-root . \
  --epochs 2 --batch 1 --grad-accum 8 --lr 5e-5 --rank 16 \
  --eval-mcq 10 --eval-open 60 --eval-every 200 --save-every 200
