#!/usr/bin/env bash
#SBATCH --job-name=cayn-medsweep-eval
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.out
# Eval the med-volume sweep on the 300-val (spec §6.5.30 follow-up).
# Points: leader (rendered-anno, 0.9548) | med4k (0.9617) | med1k | med7k.
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
GT="$WS/data/processed/annotated_samples/OCR/validation"
OUT="$WS/vit_medsweep_eval"; mkdir -p "$OUT"

trans () {
  local name="$1" mdir="$2"; [ -z "$mdir" ] && { echo "skip $name (no model)"; return 0; }
  python scripts/ocr/run_trocr_transcribe.py --model-dir "$mdir" --input-dir "$GT" \
    --output-dir "$OUT/${name}_pred" --run-name "$name" --device cuda \
    --resize-mode stretch --num-beams 4 || return 0
}
trans "leader_0.9548"        "$WS/models/mixed_med4k_fixed/trocr_20260814_173955/best_model"
trans "vit_lightreal_med4k"  "$(ls -d $WS/models/vit_lightreal_med4k/trocr_*/best_model 2>/dev/null | head -1)"
trans "vit_lightreal_med1k"  "$(ls -d $WS/models/vit_lightreal_med1k/trocr_*/best_model 2>/dev/null | head -1)"
trans "vit_lightreal_med7k"  "$(ls -d $WS/models/vit_lightreal_med7k/trocr_*/best_model 2>/dev/null | head -1)"

args=(); for d in "$OUT"/*_pred; do n=$(basename "$d" _pred); p="$d/$n"; [ -d "$p" ] && args+=(--pred "$n=$p"); done
python scripts/ocr/run_evaluate_ocr.py --gt-dir "$GT" "${args[@]}" \
  --output-dir "$OUT/eval" --run-name vit_medsweep
echo "DONE"; cat "$OUT/eval/vit_medsweep/vit_medsweep.md" 2>/dev/null || true
