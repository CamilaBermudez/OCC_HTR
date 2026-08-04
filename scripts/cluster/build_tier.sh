#!/usr/bin/env bash
# cayn build_tier <TIER> <FONT> — assemble one training tier by symlinking the
# medical + annotated pool PNGs into a single folder and merging their
# labels.json. Prints the tier folder path on stdout. DRAFT (2026-08-04).
#
# Same idea as the old VM orchestrator's build_tier (spec §6.5.21). Tier sizes
# (medical + annotated, 3:4-style ratio) from spec §6.5.21:
#   T2 = 12k + 9k   T3 = 36k + 27k   T4 = 120k + 90k
# Pools are regenerated on the cluster by generate_pool_set.sh (seed 42) into
# $WS/pools/aug_{medical,anno}_<N>_<font>_<DATE>/  (each with a labels.json).
set -euo pipefail
source "$(dirname "$0")/env.sh"
TIER="$1"; FONT="$2"   # FONT = 1font | mf
case "$TIER" in
  T2) MED=12000; ANNO=9000 ;;
  T3) MED=36000; ANNO=27000 ;;
  T4) MED=120000; ANNO=90000 ;;
  *) echo "unknown tier '$TIER'" >&2; exit 1 ;;
esac

OUT="$WS/pools/aug_${TIER}_${FONT}"
if [ -f "$OUT/labels.json" ]; then echo "$OUT"; exit 0; fi   # already assembled

med=$(ls -d "$WS"/pools/aug_medical_${MED}_${FONT}_* 2>/dev/null | head -1)
anno=$(ls -d "$WS"/pools/aug_anno_${ANNO}_${FONT}_* 2>/dev/null | head -1)
[ -d "$med" ] && [ -d "$anno" ] || {
  echo "missing pools for $TIER $FONT (med=$med anno=$anno) — regenerate first" >&2
  exit 1
}

mkdir -p "$OUT"
# batched symlink (ln -sft) via xargs so the 120k-file tiers don't blow ARG_MAX
find "$med" "$anno" -maxdepth 1 -name '*.png' -print0 | xargs -0 ln -sft "$OUT"
# merge labels.json (union; disjoint stems, so no collisions)
python - "$med/labels.json" "$anno/labels.json" "$OUT/labels.json" <<'PY'
import json, sys
merged = json.load(open(sys.argv[1], encoding="utf-8"))
merged.update(json.load(open(sys.argv[2], encoding="utf-8")))
json.dump(merged, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False)
PY
echo "$OUT"
