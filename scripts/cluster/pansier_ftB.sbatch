#!/usr/bin/env bash
#SBATCH --job-name=cayn-panB
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
# Approach B (pansier sweep, spec §6.5.29): warm-start from the deployed TrOCR leader
# (mixed_med4k_fixed = trocr-base-handwritten, stretch + BPE-150, 0.9549 char-acc) and
# FURTHER fine-tune on N pansier renders + the 600 real (anchor against forgetting).
# Tokenizer is inherited from the leader checkpoint (BPE-150) — do NOT override it.
#   N=300|600|1000|2000   (pansier subsample via --max-aug-samples)
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
DATA="$WS/data/processed"
N="${N:-2000}"
NAME="${NAME:-pansierB_${N}}"
LEADER="$WS/models/mixed_med4k_fixed/trocr_20260814_173955/best_model"
POOL="$DATA/synthetic_samples/augmented_images/pansier_aug_2000_20260820"
LABELS="$DATA/synthetic_samples/img_labels/labels_pansier_2000_20260820/labels.json"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} N=$N node=$(hostname)" > "$STAT/$NAME.status"

python scripts/ocr/run_trocr_finetune.py \
  --pretrained-model-id "$LEADER" \
  --real-folder "$DATA/annotated_samples/OCR/full_annotated" \
  --augmented-folder "$POOL" \
  --labels-json "$LABELS" \
  --max-aug-samples "$N" \
  --output-base-dir "$WS/models/$NAME" \
  --val-fraction 0.2 --epochs 15 --batch-size 64 --device cuda \
  --resize-mode stretch
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $WS/models/$NAME" > "$STAT/$NAME.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/$NAME.status"
exit $rc
