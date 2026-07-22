#!/bin/bash
#SBATCH --job-name=rag-score
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
python score_open_rag.py \
  --val ft_data_v3/val.jsonl \
  --model models/Qwen2.5-VL-merged \
  --index dino_index \
  --dino facebook/dinov3-vitl16-pretrain-lvd1689m \
  --k 3 --sim-floor 0.5 --n 202
