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
| pansierA_1000 | 299 | 0.0463 | 0.9537 | 0.2406 | 0.7594 | 0.0294 | 0.2000 | 0 |
| pansierA_2000 | 299 | 0.0493 | 0.9507 | 0.2460 | 0.7540 | 0.0303 | 0.2000 | 0 |
| pansierA_300 | 299 | 0.0482 | 0.9518 | 0.2455 | 0.7545 | 0.0294 | 0.2222 | 0 |
| pansierA_600 | 299 | 0.0445 | 0.9555 | 0.2382 | 0.7618 | 0.0286 | 0.2000 | 0 |
| pansierB_1000 | 299 | 0.0461 | 0.9539 | 0.2368 | 0.7632 | 0.0286 | 0.2000 | 0 |
| pansierB_2000 | 299 | 0.0446 | 0.9554 | 0.2348 | 0.7652 | 0.0294 | 0.2000 | 0 |
| pansierB_300 | 299 | 0.0451 | 0.9549 | 0.2445 | 0.7555 | 0.0294 | 0.2000 | 0 |
| pansierB_600 | 299 | 0.0454 | 0.9546 | 0.2275 | 0.7725 | 0.0286 | 0.2000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
