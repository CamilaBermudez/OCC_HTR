# Corrected line pool — provenance

Hand-verified `<stem>.png` + `<stem>.gt.txt` line pairs accumulated
across six annotation batches. Used as ground truth for the OCR
benchmark (`scripts/ocr/run_evaluate_ocr.py`) and as a seed pool for
fine-tuning the catmus base model.

**Location note (history):**

- 2026-07-02: moved from `tests/ocr/real_corrected_20260625/` (git-tracked
  under `tests/`) into `data/processed/annotated_samples/OCR/500_samples/`
  (git-tracked via a targeted gitignore exception), alongside the
  layout retrain data.
- 2026-07-05: renamed from `500_samples/` to `full_annotated/`. The
  original name was tied to a specific count that quickly stopped
  matching reality as the pool grew (500 -> 600 -> …). The new name
  is count-agnostic and contrasts with the per-batch archive folders
  (`batch_5/`, `batch_6/`, …) sitting beside it.

The `tests/ocr/` folder still holds the per-batch held-out sample
folders (`real_val_sample_*/`), each of which keeps its `_INDEX.csv`
as the canonical stem list for that batch.

## Composition (as of 2026-07-05) — 600 lines

| Batch | Lines | Merged on  | Source folder & index                                          |
|-------|-------|------------|----------------------------------------------------------------|
| 1+2   | ~200  | (initial)  | (earlier batches, no separate index kept)                      |
| 3     | 100   | 2026-06-25 | `tests/ocr/real_val_sample_20260625_*` (merge commit 106efec)  |
| 4     | 100   | 2026-06-29 | `tests/ocr/real_val_sample_20260629_175815/_INDEX.csv`         |
| 5     | 100   | 2026-07-01 | `tests/ocr/real_val_sample_20260701_213000/_INDEX.csv`         |
| 6     | 100   | 2026-07-05 | `tests/ocr/real_val_sample_20260703_100000/_INDEX.csv` (C/E targeted) |

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
