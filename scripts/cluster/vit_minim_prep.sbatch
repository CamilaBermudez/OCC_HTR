#!/usr/bin/env bash
#SBATCH --job-name=cayn-vitminim-prep
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out
# Build the ViT oversampling pool (spec §6.5.30 / §6.8.1 follow-up): take the 177
# annotated lines dense (>=2) in the issues §6.8.1 found for ViT+RoBERTa (minims
# n/m/u/i/r + dropped abbreviation tildes), make 3 EXTRA light-aug copies of each
# (seed 142, distinct from the pool's seed-42 aug00-04), rename them _aug1N so the
# source stem is preserved (= the real line) — the ViT split groups augs by source
# stem, so a hard line in val keeps its copies in val (leakage-safe, no prefix).
# Merge onto the med4k pool (poolLR, 7000) -> a new oversampled pool.
set -euo pipefail
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
source "$WS/.venv-poolgen/bin/activate"
cd "$PROJECT_ROOT"
D="$WS/data/processed"
FA="$D/annotated_samples/OCR/full_annotated"
STEMS="$D/vit_hard_stems.json"
AI="$D/synthetic_samples/augmented_images"
LB="$D/synthetic_samples/img_labels"
POOLLR="$AI/poolLR_light3k_med4k_20260823"
POOLLR_LAB="$LB/labels_poolLR_light3k_med4k_20260823/labels.json"

# 1. gather the 177 ViT-hard real crops
HARD="$AI/vit_hard_crops"; rm -rf "$HARD"; mkdir -p "$HARD"
python3 - "$STEMS" "$FA" "$HARD" <<'PY'
import json, os, shutil, sys
stems = json.load(open(sys.argv[1]))["vit_hard_ge2"]; fa, dst = sys.argv[2], sys.argv[3]
n = 0
for s in stems:
    png, gt = os.path.join(fa, s + ".png"), os.path.join(fa, s + ".gt.txt")
    if os.path.exists(png) and os.path.exists(gt):
        shutil.copy(png, os.path.join(dst, s + ".png"))
        shutil.copy(gt, os.path.join(dst, s + ".gt.txt")); n += 1
print("gathered ViT-hard crops:", n)
PY

# 2. 3 EXTRA light-aug copies, seed 142 (distinct from the pool's aug00-04)
python scripts/data_augmentation/augment_real_light.py \
  --real-folder "$HARD" --output-folder "$AI/vit_hard_raw" \
  --n-augmentations 3 --seed 142

# 3. rename _aug0N -> _aug1N (keeps source stem = real line) and merge onto poolLR
OVSPOOL="$AI/poolLR_light3k_med4k_vitovs_20260824"
OVSLAB="$LB/labels_poolLR_light3k_med4k_vitovs_20260824/labels.json"
mkdir -p "$(dirname "$OVSLAB")"
python3 - "$POOLLR" "$POOLLR_LAB" "$AI/vit_hard_raw" "$OVSPOOL" "$OVSLAB" <<'PY'
import json, os, sys
poollr, poollr_lab, hardraw, outdir, outlab = sys.argv[1:6]
os.makedirs(outdir, exist_ok=True)
merged = json.load(open(poollr_lab))          # all 7000 med4k-pool labels
for name in list(merged):                     # symlink every poolLR png
    s = os.path.abspath(os.path.join(poollr, name)); d = os.path.join(outdir, name)
    if os.path.islink(d) or os.path.exists(d): os.remove(d)
    os.symlink(s, d)
hl = json.load(open(os.path.join(hardraw, "labels.json"))); added = 0
for name, txt in hl.items():                   # <stem>_aug0N.png -> <stem>_aug1N.png
    new = name.replace("_aug0", "_aug1")
    s = os.path.abspath(os.path.join(hardraw, name)); d = os.path.join(outdir, new)
    if os.path.islink(d) or os.path.exists(d): os.remove(d)
    os.symlink(s, d); merged[new] = txt; added += 1
json.dump(merged, open(outlab, "w"), ensure_ascii=False)
print(f"ovs pool total {len(merged)} (poolLR 7000 + hard-extra {added})")
PY
echo "PREP DONE $(date -Is)"
