# OCR evaluation — paddleocr_vs_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `catmus` from `data/processed/transcription/ocr_kept_20260622_120413`
- `paddleocr_latin` from `data/processed/transcription/paddleocr_latin_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| catmus | 299 | 0.0397 | 0.9603 | 0.1488 | 0.8512 | 0.0278 | 0.1429 | 0 |
| paddleocr_latin | 299 | 0.2328 | 0.7672 | 0.6636 | 0.3364 | 0.2069 | 0.6667 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
