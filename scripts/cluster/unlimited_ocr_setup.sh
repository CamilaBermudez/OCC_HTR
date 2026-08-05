#!/usr/bin/env bash
# cayn — build the Unlimited-OCR (Baidu VLM / DeepSeek-OCR base) env + prefetch
# weights. Separate venv (py3.12, torch 2.10, transformers 4.57.1) — a distinct
# stack from the TrOCR (2.13) and kraken (2.4.1) venvs. Downloads via the TF proxy.
#
# Run ONCE on a GPU node:
#   srun -p testdlc2_gpu-h200 --gres=gpu:1 --pty bash
#   bash ~/cayn/scripts/cluster/unlimited_ocr_setup.sh
set -euo pipefail
export CAYN_NO_VENV=1
source "$(dirname "$0")/env.sh"
export UV_UOCR="$WS/.venv-uocr"

uv venv --python 3.12 "$UV_UOCR"
uv pip install --python "$UV_UOCR/bin/python" \
    torch==2.10.0 torchvision==0.25.0 transformers==4.57.1 \
    "pillow>=11" einops accelerate safetensors huggingface_hub

# prefetch the model into HF_HOME (on /work) so the sbatch run is offline-fast
"$UV_UOCR/bin/python" - <<'PY'
import torch, transformers
from huggingface_hub import snapshot_download
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| transformers", transformers.__version__)
p = snapshot_download("baidu/Unlimited-OCR")
print("model snapshot at", p)
PY
echo "UOCR_VENV_DONE -> $UV_UOCR"
