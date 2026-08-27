# OCR evaluation — vit_augamount

Ground truth: `/work/dlc2workfs3/zehlet-cayn/data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `vit_lx3_m2k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx3_m2k_pred/vit_lx3_m2k`
- `vit_lx3_m4k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx3_m4k_pred/vit_lx3_m4k`
- `vit_lx3_m8k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx3_m8k_pred/vit_lx3_m8k`
- `vit_lx5_m2k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx5_m2k_pred/vit_lx5_m2k`
- `vit_lx5_m4k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx5_m4k_pred/vit_lx5_m4k`
- `vit_lx5_m8k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx5_m8k_pred/vit_lx5_m8k`
- `vit_lx7_m2k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx7_m2k_pred/vit_lx7_m2k`
- `vit_lx7_m4k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx7_m4k_pred/vit_lx7_m4k`
- `vit_lx7_m8k` from `/work/dlc2workfs3/zehlet-cayn/vit_augamount_eval/vit_lx7_m8k_pred/vit_lx7_m8k`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| vit_lx3_m2k | 299 | 0.0407 | 0.9593 | 0.2265 | 0.7735 | 0.0278 | 0.2000 | 0 |
| vit_lx3_m4k | 299 | 0.0397 | 0.9603 | 0.2207 | 0.7793 | 0.0278 | 0.2000 | 0 |
| vit_lx3_m8k | 299 | 0.0402 | 0.9598 | 0.2124 | 0.7876 | 0.0278 | 0.1667 | 0 |
| vit_lx5_m2k | 299 | 0.0390 | 0.9610 | 0.2134 | 0.7866 | 0.0270 | 0.1667 | 0 |
| vit_lx5_m4k | 299 | 0.0421 | 0.9579 | 0.2251 | 0.7749 | 0.0278 | 0.2000 | 0 |
| vit_lx5_m8k | 299 | 0.0366 | 0.9634 | 0.2105 | 0.7895 | 0.0278 | 0.1667 | 0 |
| vit_lx7_m2k | 299 | 0.0395 | 0.9605 | 0.2110 | 0.7890 | 0.0270 | 0.1667 | 0 |
| vit_lx7_m4k | 299 | 0.0383 | 0.9617 | 0.2110 | 0.7890 | 0.0278 | 0.1667 | 0 |
| vit_lx7_m8k | 299 | 0.0380 | 0.9620 | 0.2183 | 0.7817 | 0.0270 | 0.1667 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
