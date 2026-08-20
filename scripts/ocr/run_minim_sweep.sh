#!/usr/bin/env bash
# Targeted-minim synthetic sweep (spec §6.5.22 retry). For each pool size, two recipes:
#   A: catmus + 600 real + N minim-synth  (mixed; synth unrouted -> train, real gives val)
#   B: kraken 0.9710 + N minim-synth only (further fine-tune; val carved from synth)
# 1 render/line, ketos --augment at train time. Evals each on the 300-val (raw kraken).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
export PROJECT_ROOT=.

BASE=data/processed/synthetic_samples/minim_sweep_20260817
CATMUS=models/ocr/catmus-medieval.mlmodel
KRAKEN=models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel
REAL=data/processed/annotated_samples/OCR/full_annotated
VAL=data/processed/annotated_samples/OCR/validation
OUT=models/ocr/finetuned/minim_sweep
RES=tests/ocr/evaluations/minim_sweep_20260820.txt
mkdir -p "$OUT"
echo "=== minim sweep $(date -Is) ===" > "$RES"

train_and_eval() {
  local run=$1; shift
  local outdir="$OUT/$run"
  echo ">>> TRAIN $run $(date -Is)"
  uv run python scripts/ocr/run_finetune_ocr.py "$@" \
    --augmented-folder "$BASE/${run#*_}/images" --labels-json "$BASE/${run#*_}/labels.json" \
    --augment --epochs 30 --output-base-dir "$outdir" >"$OUT/${run}_train.log" 2>&1
  local model
  model=$(ls -dt "$outdir"/finetune_*/model_best.mlmodel 2>/dev/null | head -1)
  if [ -z "$model" ]; then echo "$run: NO MODEL (see ${run}_train.log)" | tee -a "$RES"; return; fi
  echo ">>> EVAL $run"
  uv run python scripts/ocr/run_transcribe_line_crops.py --input-dir "$VAL" \
    --model-path "$model" --run-name "sweep_${run}_val300" >"$OUT/${run}_eval.log" 2>&1
  uv run python scripts/ocr/run_evaluate_ocr.py --gt-dir "$VAL" \
    --pred "sw=data/processed/transcription/sweep_${run}_val300" \
    --run-name "sweep_${run}_val300" 2>>"$OUT/${run}_eval.log" \
    | grep -E "^\| sw " | sed "s/^/$run  /" | tee -a "$RES"
}

for N in 50 100 300 600 1000; do
  train_and_eval "A_n${N}" --base-model "$CATMUS" --real-folder "$REAL" \
    --aug-unrouted-to-train --val-fraction 0.2
  train_and_eval "B_n${N}" --base-model "$KRAKEN" --val-fraction 0.15
done
echo "=== SWEEP DONE $(date -Is) ===" | tee -a "$RES"
echo "baseline: kraken 600+ketos-aug = 0.9710 (no synth)" >> "$RES"
