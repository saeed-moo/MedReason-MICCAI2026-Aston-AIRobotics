#!/bin/bash
#SBATCH --job-name=dino-index
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
python build_dino_index.py \
  --json data/train/medreason_train_selection.json \
  --imgs data/train/imgs \
  --out dino_index \
  --model facebook/dinov3-vitl16-pretrain-lvd1689m \
  --batch 32
