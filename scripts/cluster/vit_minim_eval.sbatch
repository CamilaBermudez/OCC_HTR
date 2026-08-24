#!/usr/bin/env bash
#SBATCH --job-name=cayn-vitminim-eval
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
# Eval the ViT minim/abbrev-oversampled model vs the med4k leader on the 300-val
# (spec §6.5.30 / §6.8.1 follow-up). med4k (0.9617) is the A/B baseline — the ovs
# pool has the identical stem universe, so the train/val split is identical.
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
GT="$WS/data/processed/annotated_samples/OCR/validation"
OUT="$WS/vit_minim_eval"; mkdir -p "$OUT"

trans () {
  local name="$1" mdir="$2"; [ -z "$mdir" ] && { echo "no model $name"; return 0; }
  python scripts/ocr/run_trocr_transcribe.py --model-dir "$mdir" --input-dir "$GT" \
    --output-dir "$OUT/${name}_pred" --run-name "$name" --device cuda \
    --resize-mode stretch --num-beams 4 || return 0
}
trans "vit_lightreal_med4k"      "$WS/models/vit_lightreal_med4k/$(ls $WS/models/vit_lightreal_med4k 2>/dev/null | grep trocr_ | head -1)/best_model"
trans "vit_lightreal_minimovs"  "$(ls -d $WS/models/vit_lightreal_minimovs/trocr_*/best_model 2>/dev/null | head -1)"

args=()
for d in "$OUT"/*_pred; do n=$(basename "$d" _pred); p="$d/$n"; [ -d "$p" ] && args+=(--pred "$n=$p"); done
python scripts/ocr/run_evaluate_ocr.py --gt-dir "$GT" "${args[@]}" --output-dir "$OUT/eval" --run-name vit_minim
echo "DONE"; cat "$OUT/eval/vit_minim/vit_minim.md" 2>/dev/null || true
