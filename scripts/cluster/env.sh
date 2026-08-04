#!/usr/bin/env bash
# cayn — cluster environment, sourced by every job + interactive shell.
# DRAFT (2026-08-04): verify the WS path + venv on the cluster before real runs.
#
# Group convention (see spec §7.6): code lives in ~/cayn, ALL big artifacts in
# /work (home is 75 GB-quota'd). On the cluster this project is "cayn" — never
# "occ_htr".
# Workspace resolved via the cluster's workspace tool (ws_allocate put it on
# whichever /work filesystem had room — currently dlc2workfs3, not dlclarge1).
export WS="$(ws_find cayn 2>/dev/null)"
: "${WS:=/work/dlc2workfs3/zehlet-cayn}"       # fallback if ws_find is unavailable
export PROJECT_ROOT="$HOME/cayn"              # code (rsynced from the laptop)
export PATH="$HOME/.local/bin:$PATH"          # uv lives here (non-login shells)
# Internet on the cluster is ONLY via the TF proxy — a compute-node srun shell
# doesn't source ~/.bashrc, so set it explicitly here (uv/pip/hf all honour it).
export http_proxy="http://tfsquid.informatik.intra.uni-freiburg.de:8080/"
export https_proxy="$http_proxy"
export ftp_proxy="$http_proxy"
export no_proxy="informatik.privat,informatik.uni-freiburg.de,intra.informatik.uni-freiburg.de,localhost,127.0.0.1,rz.ki.privat,tf.ki.privat,tf.uni-freiburg.de,uni-freiburg.de"
export HF_HOME="$WS/hf"                       # HuggingFace model/dataset cache
export UV_CACHE_DIR="$WS/uv-cache"
export TMPDIR="$WS/tmp"
export TOKENIZERS_PARALLELISM=false
# The cluster venv is NOT the repo's editable install, so make `import src.*`
# resolve by putting the code root on the path (repo runs as `PROJECT_ROOT=.`).
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$WS"/{pools,models,hf,uv-cache,tmp,eval}
# Activate the training venv if it has been built (node_setup.sh). Use an
# if-block (not `&&`) so sourcing this file always returns 0 — otherwise a
# caller running under `set -e` (node_setup.sh) exits when the venv is absent.
# The kraken track sets CAYN_NO_VENV=1 to skip this: it runs via `uv run` in the
# pyproject env (torch 2.4.1) instead of the TrOCR venv (torch 2.13).
if [ -z "${CAYN_NO_VENV:-}" ] && [ -f "$WS/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$WS/.venv/bin/activate"
fi
