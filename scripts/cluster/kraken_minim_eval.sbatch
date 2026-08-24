#!/usr/bin/env bash
#SBATCH --job-name=cayn-minimovs-eval
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
# Eval the kraken minim-oversampling arms on the 300-val (spec §6.5.30 follow-up).
# baseline (600 real + ketos aug) vs +158-dense-minim x3 light-aug oversampled.
set -uo pipefail
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
export UV_PROJECT_ENVIRONMENT="$WS/.venv-kraken"
cd "$PROJECT_ROOT"
GT="$WS/data/processed/annotated_samples/OCR/validation"
OUT="$WS/kraken_minim_eval_8020"; mkdir -p "$OUT"

trans () {
  local name="$1"
  local mdl; mdl=$(ls -t "$WS/models/$name"/*/model_best.mlmodel 2>/dev/null | head -1)
  [ -z "$mdl" ] && mdl=$(ls -t "$WS/models/$name"/model_best.mlmodel 2>/dev/null | head -1)
  [ -z "$mdl" ] && { echo "no model for $name"; return 0; }
  echo "transcribe $name <- $mdl"
  uv run python scripts/ocr/run_transcribe_line_crops.py --model-path "$mdl" \
    --input-dir "$GT" --output-dir "$OUT" --run-name "${name}_pred" --device cuda:0 || return 0
}
trans kraken_minim_base_8020
trans kraken_minim_ovs_8020

args=()
for n in kraken_minim_base_8020 kraken_minim_ovs_8020; do
  p="$OUT/${n}_pred"; [ -d "$p" ] && args+=(--pred "$n=$p")
done
uv run python scripts/ocr/run_evaluate_ocr.py --gt-dir "$GT" "${args[@]}" \
  --output-dir "$OUT/eval" --run-name kraken_minim
echo "DONE"; cat "$OUT/eval/kraken_minim/kraken_minim.md" 2>/dev/null || true
