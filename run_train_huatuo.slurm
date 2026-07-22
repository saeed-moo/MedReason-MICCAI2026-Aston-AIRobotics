#!/bin/bash
#SBATCH --job-name=medreason-huatuo
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
python train_lora_v3.py \
  --train ft_data_v3/train.jsonl --val ft_data_v3/val.jsonl \
  --model models/HuatuoGPT-Vision-7B \
  --out lora_huatuo --imgs-root . \
  --epochs 1 --batch 1 --grad-accum 16 --lr 5e-5 --rank 16 \
  --eval-mcq 200 --eval-open 40 --eval-every 400 --save-every 400
