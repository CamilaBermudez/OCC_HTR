#!/usr/bin/env bash
#SBATCH --job-name=cayn-2stage
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
# Clean two-stage ViT+RoBERTa (spec §6.5.26): Stage-1 = synthetic-ONLY gentle-aug
# medical pretrain (stretch + BPE-150), Stage-2 = fine-tune on the 600 real, from the
# Stage-1 checkpoint. Tests whether staging beats mixing (mixed medical-4000 = 0.9545).
#   SIZE=3000|6000|12000  (Stage-1 synthetic subsample)
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
SIZE="${SIZE:-12000}"
NAME="twostage_${SIZE}"
DATA="$WS/data/processed"
TOK="$DATA/tokenizer/occitan_char_bpe_150_20260806/hf_tokenizer"
POOL="$DATA/synthetic_samples/augmented_images/${POOLNAME:-med_stage1_12k_gentle_20260814}"
LABELS="$DATA/synthetic_samples/img_labels/${STAGE1_LABELS}/labels.json"
S1="$WS/models/${NAME}_s1"
S2="$WS/models/${NAME}_s2"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} size=$SIZE node=$(hostname)" > "$STAT/$NAME.status"

echo "=== Stage 1: synthetic-only pretrain ($SIZE, gentle, stretch+BPE-150) ==="
python scripts/ocr/run_trocr_finetune.py \
  --no-real \
  --augmented-folder "$POOL" --labels-json "$LABELS" --max-aug-samples "$SIZE" \
  --pretrained-model-id microsoft/trocr-base-handwritten --tokenizer "$TOK" \
  --resize-mode stretch --val-fraction 0.05 --epochs 15 --batch-size 64 --device cuda \
  --output-base-dir "$S1"
S1MODEL=$(ls -d "$S1"/trocr_*/best_model 2>/dev/null | head -1)
[ -z "$S1MODEL" ] && { echo "FAILED $(date -Is) no stage1 model" > "$STAT/$NAME.status"; exit 1; }

echo "=== Stage 2: fine-tune on 600 real, from $S1MODEL ==="
python scripts/ocr/run_trocr_finetune.py \
  --pretrained-model-id "$S1MODEL" \
  --real-folder "$DATA/annotated_samples/OCR/full_annotated" \
  --resize-mode stretch --val-fraction 0.2 --epochs 15 --batch-size 64 --device cuda \
  --output-base-dir "$S2"
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $S2" > "$STAT/$NAME.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/$NAME.status"
exit $rc
