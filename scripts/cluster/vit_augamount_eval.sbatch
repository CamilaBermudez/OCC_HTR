#!/usr/bin/env bash
#SBATCH --job-name=cayn-augamount-eval
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
# Eval the 3x3 light-aug x medical grid on the 300-val (spec §6.5.30 follow-up).
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
GT="$WS/data/processed/annotated_samples/OCR/validation"
OUT="$WS/vit_augamount_eval"; mkdir -p "$OUT"

trans () {
  local name="$1" mdir="$2"; [ -z "$mdir" ] && { echo "skip $name"; return 0; }
  python scripts/ocr/run_trocr_transcribe.py --model-dir "$mdir" --input-dir "$GT" \
    --output-dir "$OUT/${name}_pred" --run-name "$name" --device cuda \
    --resize-mode stretch --num-beams 4 || return 0
}
for k in 3 5 7; do for m in 2k 4k 8k; do
  n="vit_lx${k}_m${m}"
  trans "$n" "$(ls -d $WS/models/$n/trocr_*/best_model 2>/dev/null | head -1)"
done; done

args=()
for d in "$OUT"/*_pred; do n=$(basename "$d" _pred); p="$d/$n"; [ -d "$p" ] && args+=(--pred "$n=$p"); done
python scripts/ocr/run_evaluate_ocr.py --gt-dir "$GT" "${args[@]}" --output-dir "$OUT/eval" --run-name vit_augamount
echo "DONE"; cat "$OUT/eval/vit_augamount/vit_augamount.md" 2>/dev/null || true
