#!/usr/bin/env bash
#SBATCH --job-name=cayn-minimovs-prep
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out
# Build the minim-oversample pool for kraken (spec §6.5.30 follow-up / user idea):
# take the 158 REAL annotated lines dense in minim patterns (>=2), light-augment
# each x3, and prefix them 'ovs_' so run_finetune_ocr routes ALL copies to TRAIN
# (their source stem no longer matches any real line, so --aug-unrouted-to-train
# sends them to train instead of following the source line's split). ketos
# --augment adds fresh online augmentation on top at train time.
set -euo pipefail
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
source "$WS/.venv-poolgen/bin/activate"
cd "$PROJECT_ROOT"
D="$WS/data/processed"
FA="$D/annotated_samples/OCR/full_annotated"
STEMS="$D/minim_dense_stems.json"
AI="$D/synthetic_samples/augmented_images"

# 1. gather the dense-minim real crops (png + gt) into a working folder
DENSE="$AI/minim_dense_crops"; rm -rf "$DENSE"; mkdir -p "$DENSE"
python3 - "$STEMS" "$FA" "$DENSE" <<'PY'
import json, os, shutil, sys
stems = json.load(open(sys.argv[1]))["dense_ge2"]; fa, dst = sys.argv[2], sys.argv[3]
n = 0
for s in stems:
    png, gt = os.path.join(fa, s + ".png"), os.path.join(fa, s + ".gt.txt")
    if os.path.exists(png) and os.path.exists(gt):
        shutil.copy(png, os.path.join(dst, s + ".png"))
        shutil.copy(gt, os.path.join(dst, s + ".gt.txt")); n += 1
print("gathered dense crops:", n)
PY

# 2. light-augment x3 (mild affine + light blur/noise, no composite/aging)
python scripts/data_augmentation/augment_real_light.py \
  --real-folder "$DENSE" --output-folder "$AI/minim_ovs_raw" \
  --n-augmentations 3 --seed 42

# 3. prefix 'ovs_' so every copy is unrouted -> forced to TRAIN
python3 - "$AI/minim_ovs_raw" "$AI/minim_ovs_pool" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
labels = json.load(open(os.path.join(src, "labels.json"))); out = {}
for name, txt in labels.items():
    new = "ovs_" + name
    s = os.path.abspath(os.path.join(src, name)); d = os.path.join(dst, new)
    if os.path.islink(d) or os.path.exists(d): os.remove(d)
    os.symlink(s, d); out[new] = txt
json.dump(out, open(os.path.join(dst, "labels.json"), "w"), ensure_ascii=False)
print("ovs pool (light-aug copies, prefixed):", len(out))
PY
echo "PREP DONE $(date -Is)"
