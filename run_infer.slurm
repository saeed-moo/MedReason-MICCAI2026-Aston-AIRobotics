#!/bin/bash
#SBATCH --job-name=medreason-infer
#SBATCH --account=brics.u6mn
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason

# load conda + activate the env (needed inside the batch shell too)
source "$ROOT/envs/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/medreason"

echo "=== node $(hostname) ==="
nvidia-smi

MODEL="$ROOT/MedReason-Challenge-Docker/MedReason-Docker-Example-Qwen25VL/models/Qwen2.5-VL"
VAL="$ROOT/data/validation"
OUT="$ROOT/outputs/results_${SLURM_JOB_ID}.json"

python "$ROOT/medreason_infer.py" \
    --json  "$VAL/medreason_validation_participant_facing.json" \
    --imgs  "$VAL/imgs" \
    --model "$MODEL" \
    --out   "$OUT" \
    --limit 8                 # SMOKE TEST: first 8 cases only. Remove for full run.

echo "=== done -> $OUT ==="
python - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("answers:", len(d["answers"]))
for a in d["answers"][:8]:
    print(" ", a)
PY
