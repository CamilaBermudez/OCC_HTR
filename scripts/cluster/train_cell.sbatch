#!/usr/bin/env bash
#SBATCH --job-name=cayn-vit
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# NB: no --mail-type — email goes to the user, not the assistant, so it's not
# useful for monitoring. Instead: (a) `squeue -u $USER` shows live state, and
# (b) each job writes $WS/status/<cell>.status = RUNNING / DONE / FAILED that a
# single `cat` reveals. The assistant polls both whenever the session is up.
#
# One ViT+RoBERTa cell. Submit per (tier, font), e.g.:
#   sbatch --export=ALL,TIER=T2,FONT=1font --job-name=cayn-vit-T2-1font \
#          scripts/cluster/train_cell.sbatch
set -euo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"

CELL="vitroberta_${TIER}_${FONT}"
STATUS="$WS/status"; mkdir -p "$STATUS"
STAT="$STATUS/$CELL.status"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT"
# any error / early exit -> record FAILED (DONE below clears it on success)
trap 'echo "FAILED $(date -Is)" >> "'"$STAT"'"' ERR

# assemble the tier (symlink medical+anno pools + merge labels.json)
POOL=$(bash scripts/cluster/build_tier.sh "$TIER" "$FONT")

# train: single-stage, pretrained cross-attention; big batch on the 141 GB H200
python scripts/ocr/run_trocr_finetune.py \
  --pretrained-model-id microsoft/trocr-base-handwritten \
  --augmented-folder "$POOL" \
  --labels-json "$POOL/labels.json" \
  --output-base-dir "$WS/models/$CELL" \
  --batch-size 64 --gradient-accumulation-steps 1 \
  --epochs 15 --device cuda --resize-mode pad

echo "DONE $(date -Is) -> $WS/models/$CELL" > "$STAT"
