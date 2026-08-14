#!/usr/bin/env bash
#SBATCH --job-name=cayn-realonly
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
# Real-only control (spec §6.5.27 A0): ViT+RoBERTa fine-tuned on the 600 real ONLY,
# stretch + BPE-150 (no synthetic). Isolates Stage-1's contribution in the two-stage
# comparison; the true stretch+BPE-150 real-only number (old real-only 0.9371 was
# pad + default tokenizer).
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT"
NAME="realonly_stretch_bpe"
DATA="$WS/data/processed"
TOK="$DATA/tokenizer/occitan_char_bpe_150_20260806/hf_tokenizer"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT/$NAME.status"
python scripts/ocr/run_trocr_finetune.py \
  --pretrained-model-id microsoft/trocr-base-handwritten --tokenizer "$TOK" \
  --real-folder "$DATA/annotated_samples/OCR/full_annotated" \
  --resize-mode stretch --val-fraction 0.2 --epochs 15 --batch-size 64 --device cuda \
  --output-base-dir "$WS/models/$NAME"
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is) -> $WS/models/$NAME" > "$STAT/$NAME.status" \
             || echo "FAILED $(date -Is) rc=$rc" > "$STAT/$NAME.status"
exit $rc
