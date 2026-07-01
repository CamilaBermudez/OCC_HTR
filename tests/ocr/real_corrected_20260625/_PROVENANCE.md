# Corrected line pool — provenance

Hand-verified `<stem>.png` + `<stem>.gt.txt` line pairs accumulated
across four annotation batches. Used as ground truth for the OCR
benchmark (`scripts/ocr/run_evaluate_ocr.py`) and as a seed pool for
fine-tuning the catmus base model.

## Composition (as of 2026-07-01) — 500 lines

| Batch | Lines | Merged on  | Source folder & index                                          |
|-------|-------|------------|----------------------------------------------------------------|
| 1+2   | ~200  | (initial)  | (earlier batches, no separate index kept)                      |
| 3     | 100   | 2026-06-25 | `tests/ocr/real_val_sample_20260625_*` (merge commit 106efec)  |
| 4     | 100   | 2026-06-29 | `tests/ocr/real_val_sample_20260629_175815/_INDEX.csv`         |
| 5     | 100   | 2026-07-01 | `tests/ocr/real_val_sample_20260701_213000/_INDEX.csv`         |

Per-batch indexes (batch 4 and 5) remain in their original held-out
folders as the canonical stem lists — use them to filter this pool back
down to a specific batch when reporting per-batch metrics or comparing
Medusa (first run only on the batch-4 lines) against catmus and the
fine-tuned model.

## Conventions
- `.gt.txt` files are NORMALISED: plain `s` not `ſ`, plain `r` not `ꝛ`,
  `et` not `⁊`. Verbatim OCR outputs need normalising before CER
  comparison.
- One line per `.gt.txt` file; some may be empty (skip those for CER).
- Stems include the page tag (e.g. `08_f_003v_004_line_50`) so a stem
  alone is enough to look up which folio it came from.
