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
# Unlimited-OCR over the 300-val line crops. Run as SBATCH (survives ssh/VPN
# drops — unlike interactive srun). LIMIT=N runs a probe over the first N crops
# (out-dir uocr_probe); unset LIMIT runs the full set. PROMPT overridable.
# Read results from the job's %x_%j.out (captured on the cluster) + the pred dir.
set -uo pipefail
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
NAME="uocr${LIMIT:+_probe}"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT/$NAME.status"

OUT="$WS/preds/unlimited_ocr_val300"; [ -n "${LIMIT:-}" ] && OUT="$WS/preds/uocr_probe"
extra=(); [ -n "${LIMIT:-}" ] && extra+=(--limit "$LIMIT")
# only override the prompt if PROMPT is set — else the script's own default
# ("<image>\nOCR:", a REAL newline) applies. Passing it here would inject a
# bash-literal backslash-n and the empty-yielding "Free OCR." prompt.
[ -n "${PROMPT:-}" ] && extra+=(--prompt "$PROMPT")

"$WS/.venv-uocr/bin/python" scripts/ocr/unlimited_ocr_transcribe.py \
  --input-dir data/processed/annotated_samples/OCR/validation \
  --out-dir "$OUT" "${extra[@]}"
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $OUT" > "$STAT/$NAME.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/$NAME.status"
exit $rc
