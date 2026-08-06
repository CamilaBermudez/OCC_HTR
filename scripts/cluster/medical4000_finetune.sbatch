#!/usr/bin/env bash
#SBATCH --job-name=cayn-med4k
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
# ViT+RoBERTa on the medical-4000 dataset (600 real + 3000 anno-aug + 4000
# medical-aug), 80/20 by source stem. Reruns of medical-4000 (0.9487, trained with
# stretch) with the resize + tokenizer knobs (spec §6.5.22):
#   TAG=pad   RESIZE=pad                              -> Run A
#   TAG=pad_tok RESIZE=pad TOKENIZER=<hf_tokenizer>   -> Run B (custom BPE-150)
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
NAME="vitroberta_medical4000_${TAG:-pad}"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT/$NAME.status"

DATA="$WS/data/processed"
extra=(); [ -n "${TOKENIZER:-}" ] && extra+=(--tokenizer "$TOKENIZER")

python scripts/ocr/run_trocr_finetune.py \
  --pretrained-model-id microsoft/trocr-base-handwritten \
  --real-folder "$DATA/annotated_samples/OCR/full_annotated" \
  --augmented-folder "$DATA/synthetic_samples/augmented_images/aug_20260723_v3_medical_4000" \
  --labels-json "$DATA/synthetic_samples/img_labels/labels_20260723_v3_medical_4000/labels.json" \
  --output-base-dir "$WS/models/$NAME" \
  --val-fraction 0.2 --epochs 15 --batch-size 64 --device cuda \
  --resize-mode "${RESIZE:-pad}" "${extra[@]}"
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $WS/models/$NAME" > "$STAT/$NAME.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/$NAME.status"
exit $rc
