# OCR evaluation — vit_minim

Ground truth: `/work/dlc2workfs3/zehlet-cayn/data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `vit_lightreal_med4k` from `/work/dlc2workfs3/zehlet-cayn/vit_minim_eval/vit_lightreal_med4k_pred/vit_lightreal_med4k`
- `vit_lightreal_minimovs` from `/work/dlc2workfs3/zehlet-cayn/vit_minim_eval/vit_lightreal_minimovs_pred/vit_lightreal_minimovs`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| vit_lightreal_med4k | 299 | 0.0383 | 0.9617 | 0.2173 | 0.7827 | 0.0278 | 0.1818 | 0 |
| vit_lightreal_minimovs | 299 | 0.0388 | 0.9612 | 0.2154 | 0.7846 | 0.0278 | 0.2000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
