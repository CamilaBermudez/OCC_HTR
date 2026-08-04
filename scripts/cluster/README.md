# cayn — KI-Cluster deployment (DRAFT 2026-08-04)

Resume the ViT+RoBERTa **T2–T4** grid on the Freiburg TF KI-Cluster. On the
cluster this project is **`cayn`** — never write `occ_htr` there. Login / VPN /
passwords + the *no-modify* access policy are in the gitignored
`spec_server_connection.md`; the resource + convention notes are in spec §7.6.

Conventions followed: **code in `~/cayn`, all big artifacts in `/work`**
(`$WS` via `ws_find cayn`; currently `/work/dlc2workfs3/zehlet-cayn`), **`uv`** venv, **`sbatch`** jobs (which run
independently of any SSH session), **logs-based** tracking (no wandb).

## Files
| file | what |
|---|---|
| `env.sh` | sourced by every job: `$WS`/`$PROJECT_ROOT`/`HF_HOME`/caches + venv activate |
| `node_setup.sh` | one-time: `uv venv` + CUDA torch + `transformers==5.12.1` + deps |
| `build_tier.sh` | assemble a tier: symlink medical+anno pool PNGs + merge labels |
| `train_cell.sbatch` | one ViT+RoBERTa cell (tier × font) on an H200 |

## Order of operations
```bash
# --- from the LAPTOP (VPN up, one `ssh kislurm` session open) ---
# 1. workspace (allocate via the cluster tool) + code dir
ssh kislurm 'ws_allocate cayn 60; mkdir -p ~/cayn; ln -sfn "$(ws_find cayn)/data" ~/cayn/data'
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'models' \
  --exclude 'data/raw' --exclude 'tests/ocr/evaluations' --exclude 'frontend' \
  --exclude 'notebooks' --exclude 'spec*.md' \
  ./ kislurm:cayn/                       # -> ~/cayn  (no occ_htr remote, no docs)

# 2. pool-generation INPUTS -> $WS (small; pools are REGENERATED, not uploaded)
rsync -avz data/raw/medical_texts* data/processed/annotated_samples \
  <fonts> <glyphs> <parchment> data/processed/annotated_samples/OCR/validation \
  kislurm:/work/dlclarge1/zehlet-cayn/inputs/          # exact paths TBD on-cluster

# --- on the CLUSTER ---
# 3. environment (once, on a GPU node so CUDA torch matches the driver)
srun -p testdlc2_gpu-h200 --gres=gpu:1 --pty bash -c 'bash ~/cayn/scripts/cluster/node_setup.sh'

# 4. regenerate the T2/T3/T4 pools (seed 42) on a CPU node  -> $WS/pools/
sbatch -p mldlc2_cpu-epyc9655 ... generate_pool_set.sh   # (wrap; ~big, CPU-heavy)

# 5. SMOKE TEST first (1 h debug partition, tiny)
sbatch -p testdlc2_gpu-h200 --time=1:00:00 --export=ALL,TIER=T2,FONT=1font \
  --job-name=cayn-smoke scripts/cluster/train_cell.sbatch

# 6. full grid — 6 cells, parallel on idle H200s
for T in T2 T3 T4; do for F in 1font mf; do
  sbatch --export=ALL,TIER=$T,FONT=$F --job-name=cayn-vit-$T-$F scripts/cluster/train_cell.sbatch
done; done
```

## Monitoring (no email — assistant can't read it)
- Live state: `squeue -u $USER`
- Per-cell result: `cat $WS/status/vitroberta_*.status`   (RUNNING / DONE / FAILED)
- Progress/metrics: `tail ~/cayn-vit-*_*.out` and the training `logs/`
The assistant polls these whenever the session is up.

## Harvest (back on the LAPTOP)
`rsync` each `$WS/models/<cell>/.../best_model` down, **rename cayn→occ_htr**, run
the 300-val eval locally (MPS), log CER/WER to spec §6.5.21.

> **DRAFT** — verify on-cluster before real runs: exact input paths, that CUDA
> torch imports on an H200, and the regenerated pool folder names
> (`build_tier.sh` globs `aug_{medical,anno}_<N>_<font>_*`).
