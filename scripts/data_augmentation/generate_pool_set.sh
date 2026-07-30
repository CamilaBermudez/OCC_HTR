#!/usr/bin/env bash
# =============================================================================
# Generate the 18-pool experiment set (new render size, stamps ON, seed 42).
#
# 3 corpora x {1-font, multi-font} x several sizes = 18 augmented pools:
#   COMETA   (88,828 texts):  x3 -> 266,478                      [2 pools]
#   Medical  (12,012 texts):  4,000 / 12,012 / 36,036 / 120,120 [8 pools]
#                             (=x1-filtered-to-4k / x1 / x3 / x10)
#   Annotated(600 lines):     3,000 / 9,000 / 27,000 / 90,000    [8 pools]
#                             (=x5 / x15 / x45 / x150; 3:4 ratio to medical)
#
# New pipeline (current repo code): font_size 24 / margin 7 (~40-44px lines,
# matching the real crops), torn-edges DISABLED, degradation kernels rescaled
# (spec 6.5.18), curated 6-font pool in fonts/ (spec 6.5.19), stamps ON.
# 1-font pools use fonts/merged_font_code_cmpl2.ttf; multi-font uses --fonts-dir
# fonts/ (Missaali + Jena1330 + _aeiou2U + xenipp3U + xibern2U + merged).
#
# REPLICATE (e.g. on a VM): from the repo root, ensure the source assets exist
#   - COMETA + medical corpus JSONs (data/processed/synthetic_seeds/...)
#   - fonts/ (6 curated) + glyphs/ (stamps) + a parchment_crops run
#   - annotated seeds: rebuilt below via seeds_from_real.py from full_annotated
# then run:  bash scripts/data_augmentation/generate_pool_set.sh
# Idempotent/resumable: each render/augment step skips if its output exists.
# =============================================================================
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
RUN(){ env PROJECT_ROOT=. PYTHONPATH=. uv run python "$@"; }

# ---- source assets (edit these if paths differ on the replication host) ----
COMETA_CORPUS=data/processed/synthetic_seeds/categorize_20260613_214958/cometa_categorized.json
MEDICAL_CORPUS=data/processed/synthetic_seeds/categorize_20260625_143327/medical_texts_categorized.json
ANNO_REAL_FOLDER=data/processed/annotated_samples/OCR/full_annotated
PARCH=data/processed/synthetic_samples/parchment_crops/parchments_20260608_082718
MERGED=fonts/merged_font_code_cmpl2.ttf
FONTS_DIR=fonts
TXT=data/processed/synthetic_text
AUGB=data/processed/synthetic_samples/augmented_images
LBLB=data/processed/synthetic_samples/img_labels

# Label substitutions. NO case-folding (user decision 2026-07-31): keep only
# the u/v and i/j medieval orthography, case-preserving. The catmus default
# additionally folds uppercase {I,U,T,A,E,S,O,H,M,D,Q,F}->lowercase; we drop
# that. NB: the real GT is ~99.5% lowercase, so cased labels will mismatch it
# at eval unless the GT is re-cased — see spec 6.5.20. Labels are cheap to
# regenerate (correct_labels only), independent of the images.
SUBS="v:u,V:U,j:i,J:I"
STAMPS="--et-stamp-dir glyphs/et --c-stamp-dir glyphs/C_capitol --e-stamp-dir glyphs/E_capitol --abbrev-base-dir glyphs --enable-pattern-stamps"
RCOMMON="--font-size 24 --margin 7 --base-seed 42 --p-long-s-begin 0.95 --p-long-s-middle 0.8 --p-rotunda-r 0.7 --p-tironian-et 0.3 --p-capital-e 0.4 --p-abbreviation 0.1 --p-end-decor 0.3 --max-abbreviation-per-line 3 --max-abbreviation-per-word 1"
say(){ echo "[$(date '+%F %T')] $*"; }

# ---- annotated seeds (build once from the 600 GT) ----
ANNO_SEEDS=data/processed/synthetic_seeds/from_real_pools/seeds_from_real.json
if [ ! -f "$ANNO_SEEDS" ]; then
  say "build annotated seeds_from_real"
  RUN scripts/data_augmentation/seeds_from_real.py --real-folder "$ANNO_REAL_FOLDER" \
    --output-dir data/processed/synthetic_seeds --run-name from_real_pools || exit 1
fi

# render <run-name> <input-json> <font-flag...>
render(){ local name=$1 inp=$2; shift 2
  [ -f "$TXT/$name/labels.json" ] && { say "SKIP render $name (exists)"; return; }
  say "RENDER $name"
  RUN scripts/data_augmentation/run_medieval_text_generation.py --input-json "$inp" \
    --output-dir "$TXT" --run-name "$name" "$@" $RCOMMON $STAMPS || exit 1
}
# augment <render-name> <aug-name> <n-aug>
augment(){ local rname=$1 aname=$2 n=$3
  [ "$(ls "$AUGB/$aname"/*.png 2>/dev/null | wc -l)" -gt 0 ] && { say "SKIP aug $aname (exists)"; return; }
  say "AUGMENT $aname (x$n)"
  RUN scripts/data_augmentation/run_augment_images.py --input-folder "$TXT/$rname" \
    --parchment-folder "$PARCH" --output-folder "$AUGB" --run-name "$aname" --n-augmentations "$n" --seed 42 || exit 1
}
# label <render-name> <aug-name>  (aug filename -> corrected text; original_text field)
label(){ local rname=$1 aname=$2
  [ -f "$LBLB/labels_${aname#aug_}/labels.json" ] && { say "SKIP labels $aname (exists)"; return; }
  say "LABELS $aname"
  RUN scripts/data_augmentation/run_label_correction.py --input-json "$TXT/$rname/labels.json" \
    --augmented-folder "$AUGB/$aname" --output-base-dir "$LBLB" --text-field original_text \
    --substitutions "$SUBS" || exit 1
}
# medical 4k: random-sample 4000 renders from the x1 (12k) pool + its labels
filter4k(){ local src=$1 dst=$2
  [ "$(ls "$AUGB/$dst"/*.png 2>/dev/null | wc -l)" -gt 0 ] && { say "SKIP filter $dst (exists)"; return; }
  say "FILTER $dst (4000 random from $src)"
  RUN - "$AUGB/$src" "$AUGB/$dst" 4000 <<'PY'
import os,sys,random,shutil
src,dst,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
os.makedirs(dst,exist_ok=True)
files=sorted(f for f in os.listdir(src) if f.endswith('.png'))
random.Random(42).shuffle(files)
for f in files[:n]: shutil.copy2(os.path.join(src,f),os.path.join(dst,f))
print(f"copied {min(n,len(files))} of {len(files)}")
PY
}

for MODE in 1font mf; do
  if [ "$MODE" = "1font" ]; then FONT="--font-path $MERGED"; else FONT="--fonts-dir $FONTS_DIR"; fi
  say "===== $MODE ====="
  # COMETA x3 -> 266,478
  render "cometa_${MODE}" "$COMETA_CORPUS" $FONT
  augment "cometa_${MODE}" "aug_cometa_266k_${MODE}" 3; label "cometa_${MODE}" "aug_cometa_266k_${MODE}"
  # MEDICAL x1/x3/x10 + 4k filter
  render "medical_${MODE}" "$MEDICAL_CORPUS" $FONT
  augment "medical_${MODE}" "aug_medical_12k_${MODE}" 1;  label "medical_${MODE}" "aug_medical_12k_${MODE}"
  filter4k "aug_medical_12k_${MODE}" "aug_medical_4k_${MODE}"; label "medical_${MODE}" "aug_medical_4k_${MODE}"
  augment "medical_${MODE}" "aug_medical_36k_${MODE}" 3;  label "medical_${MODE}" "aug_medical_36k_${MODE}"
  augment "medical_${MODE}" "aug_medical_120k_${MODE}" 10; label "medical_${MODE}" "aug_medical_120k_${MODE}"
  # ANNOTATED x5/x15/x45/x150 (3:4 ratio to medical)
  render "anno_${MODE}" "$ANNO_SEEDS" $FONT
  augment "anno_${MODE}" "aug_anno_3k_${MODE}" 5;    label "anno_${MODE}" "aug_anno_3k_${MODE}"
  augment "anno_${MODE}" "aug_anno_9k_${MODE}" 15;   label "anno_${MODE}" "aug_anno_9k_${MODE}"
  augment "anno_${MODE}" "aug_anno_27k_${MODE}" 45;  label "anno_${MODE}" "aug_anno_27k_${MODE}"
  augment "anno_${MODE}" "aug_anno_90k_${MODE}" 150; label "anno_${MODE}" "aug_anno_90k_${MODE}"
done
say "ALL_18_POOLS_DONE"
