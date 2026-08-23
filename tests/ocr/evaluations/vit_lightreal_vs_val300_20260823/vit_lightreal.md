# OCR evaluation — vit_lightreal

Ground truth: `/work/dlc2workfs3/zehlet-cayn/data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `leader_0.9549` from `/work/dlc2workfs3/zehlet-cayn/vit_lightreal_eval/leader_0.9549_pred/leader_0.9549`
- `vit_lightreal_med4k` from `/work/dlc2workfs3/zehlet-cayn/vit_lightreal_eval/vit_lightreal_med4k_pred/vit_lightreal_med4k`
- `vit_lightreal_only` from `/work/dlc2workfs3/zehlet-cayn/vit_lightreal_eval/vit_lightreal_only_pred/vit_lightreal_only`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| leader_0.9549 | 299 | 0.0452 | 0.9548 | 0.2295 | 0.7705 | 0.0286 | 0.1667 | 0 |
| vit_lightreal_med4k | 299 | 0.0383 | 0.9617 | 0.2173 | 0.7827 | 0.0278 | 0.1818 | 0 |
| vit_lightreal_only | 299 | 0.0427 | 0.9573 | 0.2275 | 0.7725 | 0.0278 | 0.2000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
