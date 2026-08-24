#!/usr/bin/env bash
#SBATCH --job-name=cayn-minimovs-train
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
# Train one kraken arm from catmus with ketos online --augment (spec §6.5.30
# follow-up). Baseline vs minim-oversampled share the identical recipe so the
# Δ isolates the oversampling. Set OVS=<pool> for the oversampled arm.
#   NAME=kraken_minim_base                       -> baseline (600 real + ketos aug)
#   NAME=kraken_minim_ovs OVS=.../minim_ovs_pool -> + 158 dense x3 light-aug (train)
set -uo pipefail
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
export UV_PROJECT_ENVIRONMENT="$WS/.venv-kraken"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
: "${NAME:?set NAME}"
D="$WS/data/processed"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} name=$NAME" > "$STAT/$NAME.status"

ovs=()
if [ -n "${OVS:-}" ]; then
  ovs+=(--augmented-folder "$OVS" --labels-json "$OVS/labels.json" --aug-unrouted-to-train)
else
  ovs+=(--no-synth-train)   # baseline: real-only + ketos aug (the 0.9710 recipe)
fi

uv run python scripts/ocr/run_finetune_ocr.py \
  --base-model models/ocr/catmus-medieval.mlmodel \
  --real-folder "$D/annotated_samples/OCR/full_annotated" \
  --real-train-frac 0.6 --real-val-frac 0.4 \
  --output-base-dir "$WS/models/$NAME" \
  --val-fraction 0.1 --seed 42 \
  --lrate 1e-5 --lag 5 --resize union --epochs -1 \
  --batch-size 32 --device cuda:0 --augment "${ovs[@]}"
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $WS/models/$NAME" > "$STAT/$NAME.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/$NAME.status"
exit $rc
