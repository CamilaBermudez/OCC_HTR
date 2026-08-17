#!/usr/bin/env bash
#SBATCH --job-name=cayn-medusaconf
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.out
# Medusa 0.2 (9B VLM) per-char confidence dump over the 300-val, for the
# 'does the model know when it's wrong?' calibration (spec §6.13). Writes a JSON
# {stem: {text, confs}} (cleaned to Medusa's first non-noise line); the light
# calibration + plot are done locally after pulling the JSON.
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT" "$WS/preds"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT/medusaconf.status"
PROJECT_ROOT=. python scripts/ocr/medusa_confidence_dump.py \
  --input-dir "$WS/data/processed/annotated_samples/OCR/validation" \
  --out "$WS/preds/medusa_conf_val300.json" --device cuda
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $WS/preds/medusa_conf_val300.json" > "$STAT/medusaconf.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/medusaconf.status"
exit $rc
