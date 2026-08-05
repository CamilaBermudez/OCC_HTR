# OCR evaluation — lexcorr_sweep

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `baseline` from `data/processed/transcription/ocr_kept_20260622_120413`
- `thr93` from `data/processed/transcription/catmus_lexcorr93_val300`
- `thr96` from `data/processed/transcription/catmus_lexcorr96_val300`
- `thr99` from `data/processed/transcription/catmus_lexcorr99_val300`
- `trainGTonly90` from `data/processed/transcription/catmus_lexcorrTG_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 299 | 0.0397 | 0.9603 | 0.1488 | 0.8512 | 0.0278 | 0.1429 | 0 |
| thr93 | 299 | 0.0434 | 0.9566 | 0.1658 | 0.8342 | 0.0286 | 0.1429 | 0 |
| thr96 | 299 | 0.0397 | 0.9603 | 0.1488 | 0.8512 | 0.0278 | 0.1429 | 0 |
| thr99 | 299 | 0.0397 | 0.9603 | 0.1488 | 0.8512 | 0.0278 | 0.1429 | 0 |
| trainGTonly90 | 299 | 0.0465 | 0.9535 | 0.1774 | 0.8226 | 0.0303 | 0.1429 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
