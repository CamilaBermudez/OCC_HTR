#!/usr/bin/env bash
#SBATCH --job-name=cayn-medsweep-prep
#SBATCH --partition=mldlc2_gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=1:30:00
#SBATCH --output=%x_%j.out
# Prep the light-real + medical-volume sweep (spec §6.5.30 follow-up): render
# 3000 NEW no-stamps medical lines (disjoint-by-text from the leader's 4000),
# gentle-augment + label them, then build the med1k and med7k pools.
#   med1k = 3000 light-real + first-1000 of med4k's exact 4000 medical
#   med7k = 3000 light-real + med4k's 4000 medical + the 3000 new lines
set -euo pipefail
# Render/augment pipeline needs albumentations — use the pool-gen venv, not the
# TrOCR .venv that env.sh auto-activates (CAYN_NO_VENV=1 skips that activation).
export CAYN_NO_VENV=1
source "$HOME/cayn/scripts/cluster/env.sh"
source "$WS/.venv-poolgen/bin/activate"
cd "$PROJECT_ROOT"
D="$WS/data/processed"
AI="$D/synthetic_samples/augmented_images"
LB="$D/synthetic_samples/img_labels"
PARCH="$D/synthetic_samples/parchment_crops/parchments_20260608_082718"
SEED_JSON="$D/synthetic_seeds/medical_new3000_20260823/medical_new3000_input.json"
LR="$AI/real_light_600x5_20260822"                       # 3000 light-aug-real + labels.json
POOLLR="$AI/poolLR_light3k_med4k_20260823"               # med4k pool (3000 lr + 4000 med)
POOLLR_LAB="$LB/labels_poolLR_light3k_med4k_20260823/labels.json"

# 1. Render the 3000 new medical lines — leader's medical_raw config: NO stamps
#    (long-s 0.95/0.8 + rotunda-r 0.7 only), merged font, size 24, margin 7, seed 42.
python scripts/data_augmentation/run_medieval_text_generation.py \
  --input-json "$SEED_JSON" \
  --output-dir "$D/synthetic_text" --run-name medical_new3000_20260823 \
  --font-path fonts/merged_font_code_cmpl2.ttf --font-size 24 --margin 7 \
  --p-long-s-begin 0.95 --p-long-s-middle 0.8 --p-rotunda-r 0.7 \
  --base-seed 42
RENDER="$D/synthetic_text/medical_new3000_20260823"

# 2. Gentle augment, 1 per line, seed 42, same 175 parchments as med4k_gentle.
# run_augment_images writes to output-folder/run-name, so pass $AI + the dir name.
python scripts/data_augmentation/run_augment_images.py \
  --input-folder "$RENDER" \
  --output-folder "$AI" \
  --parchment-folder "$PARCH" \
  --gentle --n-augmentations 1 --seed 42 --run-name aug_medical_new3000_gentle_20260823
NEWAUG="$AI/aug_medical_new3000_gentle_20260823"

# 3. Correct labels (v->u, j->i diplomatic fold on original_text).
python scripts/data_augmentation/run_label_correction.py \
  --input-json "$RENDER/labels.json" \
  --augmented-folder "$NEWAUG" \
  --output-base-dir "$LB/labels_medical_new3000_gentle_20260823" \
  --text-field original_text
NEWLAB="$(find "$LB/labels_medical_new3000_gentle_20260823" -name labels.json -print -quit)"
echo "NEW medical: aug=$(ls "$NEWAUG"/*.png | wc -l)  labels=$(python3 -c "import json;print(len(json.load(open('$NEWLAB'))))")"

# 4a. med4k-medical-ONLY labels = poolLR labels minus the light-real names.
python3 - "$POOLLR_LAB" "$LR/labels.json" "$LB/med4k_medical_only_labels.json" <<'PY'
import json,sys
pool=json.load(open(sys.argv[1])); lr=set(json.load(open(sys.argv[2])))
med={k:v for k,v in pool.items() if k not in lr}
json.dump(med, open(sys.argv[3],"w"), ensure_ascii=False)
print("med4k medical-only labels:", len(med))
PY
MED4K_LAB="$LB/med4k_medical_only_labels.json"

# 4b. Build the two pools (symlinks + merged labels). Nested: med1k's 1000 are
#     the first 1000 of med4k's 4000; med7k = that 4000 + the 3000 new lines.
python scripts/cluster/build_medsweep_pool.py \
  --lightreal-folder "$LR" --lightreal-labels "$LR/labels.json" \
  --medical "$POOLLR" "$MED4K_LAB" 1000 \
  --out-folder "$AI/poolLR_light3k_med1k_20260823" \
  --out-labels "$LB/labels_poolLR_light3k_med1k_20260823/labels.json"

# The new-3000 reuse stems like AnatMondG_lNNNNN that can clash by *filename*
# with med4k's medical (whose line-indexing differs — different text, same name).
# Give them a distinct 'n3k_' namespace (text stays disjoint; filenames can't clash).
NEWAUG_PFX="$AI/aug_medical_new3000_gentle_pfx_20260823"
NEWLAB_PFX="$LB/labels_medical_new3000_gentle_pfx_20260823.json"
python3 - "$NEWAUG" "$NEWLAB" "$NEWAUG_PFX" "$NEWLAB_PFX" <<'PY'
import sys, os, json
src, lab, dst, dlab = sys.argv[1:5]
os.makedirs(dst, exist_ok=True)
labels = json.load(open(lab)); out = {}
for name, txt in labels.items():
    new = "n3k_" + name
    d = os.path.join(dst, new); s = os.path.abspath(os.path.join(src, name))
    if os.path.islink(d) or os.path.exists(d): os.remove(d)
    os.symlink(s, d); out[new] = txt
json.dump(out, open(dlab, "w"), ensure_ascii=False)
print("prefixed new medical:", len(out))
PY

python scripts/cluster/build_medsweep_pool.py \
  --lightreal-folder "$LR" --lightreal-labels "$LR/labels.json" \
  --medical "$POOLLR" "$MED4K_LAB" 4000 \
  --medical "$NEWAUG_PFX" "$NEWLAB_PFX" 3000 \
  --out-folder "$AI/poolLR_light3k_med7k_20260823" \
  --out-labels "$LB/labels_poolLR_light3k_med7k_20260823/labels.json"

echo "PREP DONE $(date -Is)"
