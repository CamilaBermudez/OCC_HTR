#!/usr/bin/env bash
#SBATCH --job-name=cayn-trocrconf
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
# TrOCR-leader (mixed_med4k_fixed) per-token confidence over the full 13,677-line corpus,
# for the Model-compare tab (spec §6.5.28). Greedy (matches mixedmed4k_full_corpus).
set -uo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"
STAT="$WS/status"; mkdir -p "$STAT" "$WS/preds"
echo "RUNNING $(date -Is) job=${SLURM_JOB_ID:-?}" > "$STAT/trocrconf.status"
PROJECT_ROOT=. python scripts/ocr/vit_transcribe_conf.py \
  --input-dir "$WS/data/filtered_kept" --model-dir "$WS/models/mixed_med4k_fixed" \
  --out-dir "$WS/preds/trocrleader_conf_fullms" --device cuda --batch-size 32 --num-beams 1
rc=$?
[ $rc -eq 0 ] && echo "DONE $(date -Is)" > "$STAT/trocrconf.status" || echo "FAILED rc=$rc" > "$STAT/trocrconf.status"
exit $rc
