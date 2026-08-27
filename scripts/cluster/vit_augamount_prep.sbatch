#!/usr/bin/env bash
#SBATCH --job-name=cayn-augamount-prep
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=0:50:00
#SBATCH --output=%x_%j.out
# 3x3 grid on the ViT leader (spec §6.5.30 follow-up): light-aug amount
# {x3,x5,x7} x medical {2000, 4000, 8000}. Render light-aug x7 ONCE (seed 42) and
# derive nested x3/x5/x7; render 1000 NEW no-stamps+gentle medical lines (n1k_,
# disjoint from the 7000 already used) so medical 8000 = 4000 (med4k) + 3000 (n3k)
# + 1000 (n1k). medical 2000 = first-2000 of the 4000 (nested subset).
set -euo pipefail
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
source "$WS/.venv-poolgen/bin/activate"
cd "$PROJECT_ROOT"
D="$WS/data/processed"
AI="$D/synthetic_samples/augmented_images"
LB="$D/synthetic_samples/img_labels"
PARCH="$D/synthetic_samples/parchment_crops/parchments_20260608_082718"

# 1. render light-aug x7 of the 600 real crops
python scripts/data_augmentation/augment_real_light.py \
  --real-folder "$D/annotated_samples/OCR/full_annotated" \
  --output-folder "$AI/real_light_600x7_20260825" \
  --n-augmentations 7 --seed 42

# 2. render + gentle-augment + label the 1000 NEW medical lines (leader medical_raw config)
python scripts/data_augmentation/run_medieval_text_generation.py \
  --input-json "$D/synthetic_seeds/medical_new1000_20260825/medical_new1000_input.json" \
  --output-dir "$D/synthetic_text" --run-name medical_new1000_20260825 \
  --font-path fonts/merged_font_code_cmpl2.ttf --font-size 24 --margin 7 \
  --p-long-s-begin 0.95 --p-long-s-middle 0.8 --p-rotunda-r 0.7 --base-seed 42
python scripts/data_augmentation/run_augment_images.py \
  --input-folder "$D/synthetic_text/medical_new1000_20260825" \
  --output-folder "$AI" --parchment-folder "$PARCH" \
  --gentle --n-augmentations 1 --seed 43 --run-name aug_medical_new1000_gentle_20260825
python scripts/data_augmentation/run_label_correction.py \
  --input-json "$D/synthetic_text/medical_new1000_20260825/labels.json" \
  --augmented-folder "$AI/aug_medical_new1000_gentle_20260825" \
  --output-base-dir "$LB/labels_medical_new1000_gentle_20260825" --text-field original_text
N1RAW="$AI/aug_medical_new1000_gentle_20260825"
N1LAB="$(find "$LB/labels_medical_new1000_gentle_20260825" -name labels.json -print -quit)"
# prefix n1k_ (distinct filename namespace)
N1PFX="$AI/aug_medical_new1000_gentle_pfx_20260825"
python3 - "$N1RAW" "$N1LAB" "$N1PFX" <<'PY'
import json, os, sys
src, lab, dst = sys.argv[1:4]; os.makedirs(dst, exist_ok=True)
out = {}
for name, txt in json.load(open(lab)).items():
    new = "n1k_" + name; s = os.path.abspath(os.path.join(src, name)); d = os.path.join(dst, new)
    if os.path.islink(d) or os.path.exists(d): os.remove(d)
    os.symlink(s, d); out[new] = txt
json.dump(out, open(os.path.join(dst, "labels.json"), "w"), ensure_ascii=False)
print("n1k medical:", len(out))
PY

# 3. build the 9 pools = nested light x{3,5,7}  x  medical {2k, 4k, 8k}
python3 - "$AI" "$LB" "$N1PFX" <<'PY'
import json, os, re, sys
ai, lb, n1pfx = sys.argv[1:4]
X7 = os.path.join(ai, "real_light_600x7_20260825")
POOLLR = os.path.join(ai, "poolLR_light3k_med4k_20260823")
NEW3K = os.path.join(ai, "aug_medical_new3000_gentle_pfx_20260823")
x7lab = json.load(open(os.path.join(X7, "labels.json")))
med4k = json.load(open(os.path.join(lb, "med4k_medical_only_labels.json")))
new3k = json.load(open(os.path.join(lb, "labels_medical_new3000_gentle_pfx_20260823.json")))
new1k = json.load(open(os.path.join(n1pfx, "labels.json")))
augidx = re.compile(r"_aug(\d+)\.png$")
med4k_sorted = sorted(med4k)
med_cols = {
    "2k": {n: (med4k[n], POOLLR) for n in med4k_sorted[:2000]},
    "4k": {n: (t, POOLLR) for n, t in med4k.items()},
    "8k": {**{n: (t, POOLLR) for n, t in med4k.items()},
           **{n: (t, NEW3K) for n, t in new3k.items()},
           **{n: (t, n1pfx) for n, t in new1k.items()}},
}
for k in (3, 5, 7):
    light = {n: (t, X7) for n, t in x7lab.items()
             if (m := augidx.search(n)) and int(m.group(1)) < k}
    for mtag, medmap in med_cols.items():
        pool = os.path.join(ai, f"poolLR_lightx{k}_med{mtag}_20260825")
        os.makedirs(pool, exist_ok=True)
        merged = {}
        for n, (t, folder) in {**light, **medmap}.items():
            d = os.path.join(pool, n); s = os.path.abspath(os.path.join(folder, n))
            if os.path.islink(d) or os.path.exists(d): os.remove(d)
            os.symlink(s, d); merged[n] = t
        labdir = os.path.join(lb, f"labels_poolLR_lightx{k}_med{mtag}_20260825")
        os.makedirs(labdir, exist_ok=True)
        json.dump(merged, open(os.path.join(labdir, "labels.json"), "w"), ensure_ascii=False)
        print(f"x{k} x med{mtag}: light={len(light)} medical={len(medmap)} total={len(merged)}")
PY
echo "PREP DONE $(date -Is)"
