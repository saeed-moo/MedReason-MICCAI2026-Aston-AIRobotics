#!/bin/bash
#SBATCH --job-name=hybrid-ctest
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"
cd "$ROOT"
python test_hybrid_container.py \
  --cases MedReason-Challenge-Docker/MedReason-Docker/test/cases.json \
  --base MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL \
  --adapter lora_medreason_v4 \
  --imgs-root MedReason-Challenge-Docker/MedReason-Docker/test \
  --out /tmp/hybrid_test_results.json
echo "===== RESULTS ====="
cat /tmp/hybrid_test_results.json
