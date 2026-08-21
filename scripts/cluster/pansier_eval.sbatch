#!/usr/bin/env bash
#SBATCH --job-name=cayn-paneval
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.out
# Evaluate every present pansier sweep model (Approach A + B, all sizes) on the
# 300-line held-out validation set (spec §6.5.29). Transcribe (stretch, beam=4) ->
# run_evaluate_ocr -> collect CER/char_acc into one table. Leader baseline = 0.9549.
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
DATA="$WS/data/processed"
GT="$DATA/annotated_samples/OCR/validation"
OUT="$WS/pansier_eval"; mkdir -p "$OUT"

# Transcribe each present model over the 300-val (skip if already transcribed —
# leader + Approach B are done from the earlier partial pass; only the A models
# and any new ones need a GPU transcription pass).
transcribe_one () {
  local name="$1" mdir="$2"
  [ -z "$mdir" ] && return 0
  local pred="$OUT/${name}_pred/${name}"
  if [ -d "$pred" ] && [ "$(ls -1 "$pred"/*.txt 2>/dev/null | wc -l)" -ge 290 ]; then
    echo "skip transcribe (cached): $name"; return 0
  fi
  python scripts/ocr/run_trocr_transcribe.py \
    --model-dir "$mdir" --input-dir "$GT" --output-dir "$OUT/${name}_pred" \
    --run-name "$name" --device cuda --resize-mode stretch --num-beams 4 || return 0
}

transcribe_one "leader_0.9549" "$WS/models/mixed_med4k_fixed/trocr_20260814_173955/best_model"
for approach in A B; do
  for N in 300 600 1000 2000; do
    m=$(ls -d "$WS/models/pansier${approach}_${N}"/trocr_*/best_model 2>/dev/null | head -1)
    transcribe_one "pansier${approach}_${N}" "$m"
  done
done

# One evaluation over every prediction folder present -> single comparison table.
args=()
for d in "$OUT"/*_pred; do n=$(basename "$d" _pred); p="$d/$n"; [ -d "$p" ] && args+=(--pred "$n=$p"); done
python scripts/ocr/run_evaluate_ocr.py --gt-dir "$GT" "${args[@]}" \
  --output-dir "$OUT/eval_full" --run-name pansier_full
echo "DONE -> $OUT/eval_full/pansier_full/pansier_full.md"
cat "$OUT/eval_full/pansier_full/pansier_full.md"
