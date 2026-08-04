#!/usr/bin/env bash
#SBATCH --job-name=cayn-pools
#SBATCH --partition=mldlc2_cpu-epyc9655
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96gb
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# Regenerate the medical+anno augmented pools (seed 42) for ViT+RoBERTa T2-T4.
# CPU-only render+augment; NO GPU. Monitor: squeue + $WS/status/pools.status.
#
#   SCOPE=small -> T2/T3 tiers (medical 4k/12k/36k + anno 3k/9k/27k, both fonts)
#   SCOPE=full  -> adds the T4 giants (medical 120k + anno 90k)
# Submit e.g.:  sbatch --export=ALL,SCOPE=small scripts/cluster/regen_pools.sbatch
set -euo pipefail
source "$HOME/cayn/scripts/cluster/env.sh"
cd "$PROJECT_ROOT"

STAT="$WS/status/pools.status"; mkdir -p "$WS/status"
echo "RUNNING $(date -Is) scope=${SCOPE:-small} job=${SLURM_JOB_ID:-?} node=$(hostname)" > "$STAT"
trap 'echo "FAILED $(date -Is)" >> "'"$STAT"'"' ERR

# use the pool-gen venv (render+augment deps), NOT uv/pyproject
export POOLGEN_PY="$WS/.venv-poolgen/bin/python"
# only medical + anno (ViT+RoBERTa is single-stage; no cometa stage-1 pool)
export TRACKS="medical anno"
export SCOPE="${SCOPE:-small}"
export MODES="${MODES:-1font mf}"

bash scripts/data_augmentation/generate_pool_set.sh

echo "DONE $(date -Is)" > "$STAT"
