#!/usr/bin/env bash
#SBATCH --job-name=cayn-medsweep-train
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
# Train one ViT+RoBERTa med-volume-sweep model — identical recipe to
# vit_lightreal_med4k (trocr-base-handwritten + BPE-150 + stretch, 600 real +
# pool, val 0.2, 15 epochs, bs 64). Parametrise with NAME + POOL + LABELS.
#   NAME=vit_lightreal_med1k POOL=.../poolLR_light3k_med1k_20260823 LABELS=.../labels.json
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
D="$WS/data/processed"
TOK="$D/tokenizer/occitan_char_bpe_150_20260806/hf_tokenizer"
: "${NAME:?set NAME}"; : "${POOL:?set POOL}"; : "${LABELS:?set LABELS}"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} name=$NAME" > "$STAT/$NAME.status"

python scripts/ocr/run_trocr_finetune.py \
  --pretrained-model-id microsoft/trocr-base-handwritten \
  --real-folder "$D/annotated_samples/OCR/full_annotated" \
  --augmented-folder "$POOL" \
  --labels-json "$LABELS" \
  --tokenizer "$TOK" \
  --output-base-dir "$WS/models/$NAME" \
  --val-fraction 0.2 --epochs 15 --batch-size 64 --device cuda \
  --resize-mode stretch
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $WS/models/$NAME" > "$STAT/$NAME.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/$NAME.status"
exit $rc
