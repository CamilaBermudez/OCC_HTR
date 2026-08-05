#!/usr/bin/env bash
#SBATCH --job-name=cayn-uocr
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
# Unlimited-OCR over the 300-val line crops. Run AFTER a --limit probe has pinned
# the infer() API + confirmed the output convention. PROMPT overridable.
set -euo pipefail
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT/uocr.status"
trap 'echo "FAILED $(date -Is)" >> "'"$STAT/uocr.status"'"' ERR

"$WS/.venv-uocr/bin/python" scripts/ocr/unlimited_ocr_transcribe.py \
  --input-dir data/processed/annotated_samples/OCR/validation \
  --out-dir "$WS/preds/unlimited_ocr_val300" \
  --prompt "${PROMPT:-<image>\nFree OCR.}"

echo "DONE $(date -Is) -> $WS/preds/unlimited_ocr_val300" > "$STAT/uocr.status"
