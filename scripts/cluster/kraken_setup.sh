#!/usr/bin/env bash
# cayn — build the KRAKEN fine-tune env. The kraken track shells out via
# `uv run python _ketos_launcher.py`, so it must run inside the repo's *pyproject*
# env (torch==2.4.1, lightning==2.4.0, kraken>=6.0.2), NOT the hand-rolled TrOCR
# venv (torch 2.13). We build that env with `uv sync` and park it on /work
# (UV_PROJECT_ENVIRONMENT) so the multi-GB CUDA torch doesn't hit the 75 GB home
# quota. The default linux torch wheel is a CUDA build (backward-compatible with
# the H200's newer driver).
#
# Run ONCE on a GPU node so torch.cuda is validated against the driver:
#   srun -p testdlc2_gpu-h200 --gres=gpu:1 --pty bash
#   bash ~/cayn/scripts/cluster/kraken_setup.sh
set -euo pipefail
export CAYN_NO_VENV=1                     # don't activate the TrOCR venv
source "$(dirname "$0")/env.sh"
export UV_PROJECT_ENVIRONMENT="$WS/.venv-kraken"
cd "$PROJECT_ROOT"

# build the pyproject env (kraken + torch 2.4.1 + lightning + coremltools ...)
uv sync

uv run python - <<'PY'
import torch, kraken
print("kraken", getattr(kraken, "__version__", "?"),
      "| torch", torch.__version__,
      "| cuda:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
# ketos CLI must resolve in this env (finetune.py calls `uv run python _ketos_launcher.py`)
uv run ketos --version
echo "KRAKEN_VENV_DONE -> $UV_PROJECT_ENVIRONMENT"
