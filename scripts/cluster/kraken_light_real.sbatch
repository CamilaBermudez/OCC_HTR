#!/usr/bin/env bash
#SBATCH --job-name=cayn-kraken-lightreal
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# spec §6.5.24 follow-up: OFFLINE light-augmented REAL crops for kraken.
# 600 real + 3000 light-aug copies of the same real crops (mild affine + light
# blur/noise, no composite/aging — catmus legibility ~0.050). ketos --no-augment
# (default) so this is a clean test of OFFLINE light-aug vs the ONLINE ketos
# --augment baseline (600 real + ketos aug = 0.9710). Same leak-fixed recipe.
# NAME=kraken_light_real_noaug (default, ketos --no-augment) or
# NAME=kraken_light_real_aug AUGMENT=1 (offline light + ketos online --augment).
set -euo pipefail
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
export UV_PROJECT_ENVIRONMENT="$WS/.venv-kraken"
cd "$PROJECT_ROOT"
NAME="${NAME:-kraken_light_real_noaug}"
STAT="$WS/status/$NAME.status"; mkdir -p "$WS/status"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT"
trap 'echo "FAILED $(date -Is)" >> "'"$STAT"'"' ERR

POOL="$WS/data/processed/synthetic_samples/augmented_images/real_light_600x5_20260822"
aug=(); [ -n "${AUGMENT:-}" ] && aug+=(--augment)

uv run python scripts/ocr/run_finetune_ocr.py \
  --base-model models/ocr/catmus-medieval.mlmodel \
  --augmented-folder "$POOL" \
  --labels-json "$POOL/labels.json" \
  --real-folder data/processed/annotated_samples/OCR/full_annotated \
  --real-train-frac 0.6 --real-val-frac 0.4 \
  --aug-unrouted-to-train \
  --output-base-dir "$WS/models/$NAME" \
  --val-fraction 0.1 --seed 42 \
  --lrate 1e-5 --lag 5 --resize union --epochs -1 \
  --batch-size 32 --device cuda:0 "${aug[@]}"

echo "DONE $(date -Is) -> $WS/models/$NAME" > "$STAT"
