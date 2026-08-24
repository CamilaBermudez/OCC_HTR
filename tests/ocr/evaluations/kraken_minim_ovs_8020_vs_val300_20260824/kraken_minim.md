# OCR evaluation — kraken_minim

Ground truth: `/work/dlc2workfs3/zehlet-cayn/data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_minim_base_8020` from `/work/dlc2workfs3/zehlet-cayn/kraken_minim_eval_8020/kraken_minim_base_8020_pred`
- `kraken_minim_ovs_8020` from `/work/dlc2workfs3/zehlet-cayn/kraken_minim_eval_8020/kraken_minim_ovs_8020_pred`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_minim_base_8020 | 299 | 0.0319 | 0.9681 | 0.1930 | 0.8070 | 0.0263 | 0.1429 | 0 |
| kraken_minim_ovs_8020 | 299 | 0.0304 | 0.9696 | 0.1842 | 0.8158 | 0.0256 | 0.1429 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
