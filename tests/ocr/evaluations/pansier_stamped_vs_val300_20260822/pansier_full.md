# OCR evaluation — pansier_full

Ground truth: `/work/dlc2workfs3/zehlet-cayn/data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `leader_0.9549` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/leader_0.9549_pred/leader_0.9549`
- `pansierA_1000` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/pansierA_1000_pred/pansierA_1000`
- `pansierA_2000` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/pansierA_2000_pred/pansierA_2000`
- `pansierA_300` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/pansierA_300_pred/pansierA_300`
- `pansierA_600` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/pansierA_600_pred/pansierA_600`
- `pansierB_1000` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/pansierB_1000_pred/pansierB_1000`
- `pansierB_2000` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/pansierB_2000_pred/pansierB_2000`
- `pansierB_300` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/pansierB_300_pred/pansierB_300`
- `pansierB_600` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval/pansierB_600_pred/pansierB_600`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| leader_0.9549 | 299 | 0.0452 | 0.9548 | 0.2295 | 0.7705 | 0.0286 | 0.1667 | 0 |
| pansierA_1000 | 299 | 0.0432 | 0.9568 | 0.2270 | 0.7730 | 0.0286 | 0.2000 | 0 |
| pansierA_2000 | 299 | 0.0443 | 0.9557 | 0.2241 | 0.7759 | 0.0286 | 0.1667 | 0 |
| pansierA_300 | 299 | 0.0436 | 0.9564 | 0.2353 | 0.7647 | 0.0286 | 0.2000 | 0 |
| pansierA_600 | 299 | 0.0446 | 0.9554 | 0.2270 | 0.7730 | 0.0286 | 0.2000 | 0 |
| pansierB_1000 | 299 | 0.0491 | 0.9509 | 0.2421 | 0.7579 | 0.0303 | 0.2000 | 0 |
| pansierB_2000 | 299 | 0.0426 | 0.9574 | 0.2246 | 0.7754 | 0.0278 | 0.2000 | 0 |
| pansierB_300 | 299 | 0.0439 | 0.9561 | 0.2275 | 0.7725 | 0.0286 | 0.1667 | 0 |
| pansierB_600 | 299 | 0.0409 | 0.9591 | 0.2246 | 0.7754 | 0.0278 | 0.2000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
