#!/usr/bin/env bash
# cayn — one-time environment build. DRAFT (2026-08-04).
#
# Run ONCE on a GPU node (so the CUDA torch wheel matches the driver):
#   srun -p testdlc2_gpu-h200 --gres=gpu:1 --pty bash
#   bash ~/cayn/scripts/cluster/node_setup.sh
#
# Deps here are TRAINING-only for the TrOCR-family (ViT+RoBERTa) — NOT the repo's
# kraken/Mac pins (torch 2.4.1 CPU/MPS), which we don't want on a CUDA H200.
# transformers==5.12.1 per spec §7.2 (5.13.x breaks TrOCR pretrained loading).
set -euo pipefail
source "$(dirname "$0")/env.sh"

uv venv --python 3.11 "$WS/.venv"
source "$WS/.venv/bin/activate"
uv pip install \
    torch \
    "transformers==5.12.1" \
    accelerate pillow rapidfuzz numpy python-dotenv "huggingface_hub[hf_transfer]"

python - <<'PY'
import torch
print("torch", torch.__version__,
      "| cuda:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
echo "SETUP_DONE — venv at $WS/.venv"
