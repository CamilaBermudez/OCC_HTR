#!/usr/bin/env bash
# cayn — cluster environment, sourced by every job + interactive shell.
# DRAFT (2026-08-04): verify the WS path + venv on the cluster before real runs.
#
# Group convention (see spec §7.6): code lives in ~/cayn, ALL big artifacts in
# /work (home is 75 GB-quota'd). On the cluster this project is "cayn" — never
# "occ_htr".
export WS=/work/dlclarge1/zehlet-cayn         # workspace (data, pools, models, cache)
export PROJECT_ROOT="$HOME/cayn"              # code (rsynced from the laptop)
export HF_HOME="$WS/hf"                       # HuggingFace model/dataset cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_CACHE_DIR="$WS/uv-cache"
export TMPDIR="$WS/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WS"/{pools,models,hf,uv-cache,tmp,eval}
# Activate the training venv if it has been built (node_setup.sh).
[ -f "$WS/.venv/bin/activate" ] && source "$WS/.venv/bin/activate"
