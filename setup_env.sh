#!/bin/bash
set -euo pipefail
ROOT=/scratch/u6mn/saeedm.u6mn/MedReason
ENVS="$ROOT/envs"
mkdir -p "$ENVS"

if [ ! -d "$ENVS/miniforge3" ]; then
    cd "$ENVS"
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    bash Miniforge3-Linux-aarch64.sh -b -p "$ENVS/miniforge3"
fi
source "$ENVS/miniforge3/etc/profile.d/conda.sh"

if [ ! -d "$ENVS/medreason" ]; then
    conda create -y -p "$ENVS/medreason" python=3.10
fi
conda activate "$ENVS/medreason"

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.49.0" accelerate qwen-vl-utils pillow safetensors

echo
echo "Environment ready:  conda activate $ENVS/medreason"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
