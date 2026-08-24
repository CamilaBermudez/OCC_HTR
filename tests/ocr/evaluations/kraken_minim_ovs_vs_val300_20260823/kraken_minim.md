# OCR evaluation — kraken_minim

Ground truth: `/work/dlc2workfs3/zehlet-cayn/data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_minim_base` from `/work/dlc2workfs3/zehlet-cayn/kraken_minim_eval/kraken_minim_base_pred`
- `kraken_minim_ovs` from `/work/dlc2workfs3/zehlet-cayn/kraken_minim_eval/kraken_minim_ovs_pred`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_minim_base | 299 | 0.0325 | 0.9675 | 0.1988 | 0.8012 | 0.0263 | 0.1429 | 0 |
| kraken_minim_ovs | 299 | 0.0310 | 0.9690 | 0.1891 | 0.8109 | 0.0256 | 0.1429 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
