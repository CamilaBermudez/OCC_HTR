#!/usr/bin/env bash
#SBATCH --job-name=cayn-kraken
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# One kraken (fine-tune CATMuS) cell — single-stage, leak-fixed split. Submit per
# (tier, font), e.g.:
#   sbatch --export=ALL,TIER=T1,FONT=1font --job-name=cayn-kraken-T1-1font \
#          scripts/cluster/kraken_cell.sbatch
# Monitoring is by squeue + $WS/status/<cell>.status (RUNNING/DONE/FAILED); no
# --mail-type (see train_cell.sbatch). ketos runs on cuda:0; catmus is a small
# CRNN so 48 GB / 1 GPU is ample.
set -euo pipefail
export CAYN_NO_VENV=1                       # use the pyproject env, not the TrOCR venv
source "$HOME/cayn/scripts/cluster/env.sh"
export UV_PROJECT_ENVIRONMENT="$WS/.venv-kraken"
cd "$PROJECT_ROOT"

CELL="kraken_${TIER}_${FONT}"
STATUS="$WS/status"; mkdir -p "$STATUS"
STAT="$STATUS/$CELL.status"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT"
trap 'echo "FAILED $(date -Is)" >> "'"$STAT"'"' ERR

# assemble the tier (medical+anno pools + merged labels) — same pools as TrOCR
POOL=$(bash scripts/cluster/build_tier.sh "$TIER" "$FONT")

# optional smoke overrides: SMOKE=1 (+ SMOKE_SIZE / SMOKE_EPOCHS), BATCH
BATCH="${BATCH:-32}"
extra=()
[ -n "${SMOKE:-}" ] && extra+=(--smoke --smoke-size "${SMOKE_SIZE:-40}" --smoke-epochs "${SMOKE_EPOCHS:-2}")

# leak-fixed recipe (spec §6.3.9): real replaces synth val (0.6 train / 0.4 val
# of the 600 annotated), catmus anchored at lrate 1e-5, early-stop via --lag 5,
# codec grown by union, epochs -1 = train until the lag plateau.
uv run python scripts/ocr/run_finetune_ocr.py \
  --base-model models/ocr/catmus-medieval.mlmodel \
  --augmented-folder "$POOL" \
  --labels-json "$POOL/labels.json" \
  --real-folder data/processed/annotated_samples/OCR/full_annotated \
  --real-train-frac 0.6 --real-val-frac 0.4 \
  --aug-unrouted-to-train \
  --output-base-dir "$WS/models/$CELL" \
  --val-fraction 0.1 --seed 42 \
  --lrate 1e-5 --lag 5 --resize union --epochs -1 \
  --batch-size "$BATCH" --device cuda:0 "${extra[@]}"

echo "DONE $(date -Is) -> $WS/models/$CELL" > "$STAT"
