#!/usr/bin/env bash
#SBATCH --job-name=cayn-vitconf
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
# ViT medical-4000 transcription + per-token confidence over the 13,677 kept crops
# (spec §7.4.1). Uses the TrOCR venv (.venv, torch 2.13). Resumable per-page.
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT/vitconf.status"

python scripts/ocr/vit_transcribe_conf.py \
  --input-dir "$WS/data/filtered_kept" \
  --model-dir "$HOME/cayn/models/ocr/finetuned/trocr_20260724_145651/best_model" \
  --out-dir "$WS/preds/vitconf" \
  --device cuda --batch-size 48 --num-beams 4
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $WS/preds/vitconf" > "$STAT/vitconf.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/vitconf.status"
exit $rc
