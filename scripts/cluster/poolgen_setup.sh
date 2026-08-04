#!/usr/bin/env bash
# cayn — build the pool-GENERATION venv (render + augment deps; NO torch).
# Kept separate from the training venv so the two dep sets don't collide.
# Run once (the submit host is fine — env.sh sets the proxy). DRAFT 2026-08-04.
set -euo pipefail
source "$(dirname "$0")/env.sh"

uv venv --python 3.11 "$WS/.venv-poolgen"
uv pip install --python "$WS/.venv-poolgen/bin/python" \
    pillow numpy opencv-python-headless albumentations matplotlib tqdm fonttools python-dotenv

"$WS/.venv-poolgen/bin/python" - <<'PY'
import PIL, numpy, cv2, albumentations, matplotlib, tqdm, fontTools
print("poolgen deps OK | cv2", cv2.__version__, "| albumentations", albumentations.__version__)
PY
echo "POOLGEN_VENV_DONE -> $WS/.venv-poolgen"
