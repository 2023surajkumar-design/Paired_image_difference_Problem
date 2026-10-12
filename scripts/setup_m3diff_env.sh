#!/usr/bin/env bash
# Isolated M3Diff environment (Linux or WSL2; upstream tested Ubuntu 18.04 / Python 3.10 / CUDA 12.1 / torch 2.2.2).
# The main notebook talks to it through workers/m3diff_worker.py via subprocess, so its pins
# (torch 2.2.2, transformers 4.40.0, bitsandbytes 0.41.0) never touch the main kernel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"
[ -d M3Diff ] || git clone https://github.com/Yuan-Liu-01/M3Diff.git
cd M3Diff
git rev-parse HEAD > "$ROOT/third_party/M3Diff.REVISION"

conda create -n m3diff python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate m3diff
pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[train,eval]"      # upstream extras; bitsandbytes 0.41.0 comes with [train] (needed for 4-bit)

# Pre-fetch weights into the HF cache (7B model: expect well over 10 GB of disk)
python -c "from huggingface_hub import snapshot_download as s; print(s('IVC-liuyuan/M3Diff'))"

echo
echo "Set in the notebook config:"
echo "  m3diff_python = \"$(which python)\""
echo "  m3diff_repo   = \"$ROOT/third_party/M3Diff\""
echo "  use_m3diff    = True"
