# Permanent validation set — provenance

Hand-verified `<stem>.png` + `<stem>.gt.txt` line pairs held out from
training. **Every future model comparison in the thesis uses this
folder as the ground-truth reference** — no more moving-target
"whichever recent batch happened to be untouched" comparisons.

## Composition (as of 2026-07-07) — 300 lines (299 non-empty)

| Batch | Lines | Merged on  | Source folder & index                                          |
|-------|-------|------------|----------------------------------------------------------------|
| val-1 | 300   | 2026-07-07 | `tests/ocr/validation_20260706_163211/_INDEX.csv`              |

- Sampled with seed=100 across 70 pages from
  `data/processed/filtered_images/20260618_160948/original/kept`.
- Excludes the 600 stems already in `full_annotated/` (batches 1-6
  as of the sample date).
- 1 stem has an empty `.gt.txt` (annotator cleared it because the
  underlying crop is a nearly-blank strip). The eval script
  auto-skips those; effective val size = 299.

## Conventions
- `.gt.txt` files are NORMALISED: plain `s` not `ſ`, plain `r` not `ꝛ`,
  `et` not `⁊`. Same convention as `full_annotated/`.
- One line per `.gt.txt` file.
- Stems include the page tag (e.g. `08_f_003v_004_line_50`) so a stem
  alone is enough to look up which folio it came from.

## How to use in an evaluation

```
PROJECT_ROOT=. uv run python scripts/ocr/run_evaluate_ocr.py \
    --gt-dir ./data/processed/annotated_samples/OCR/validation \
    --pred catmus_baseline=./data/processed/transcription/ocr_kept_20260622_120413 \
    --pred medusa=./data/processed/transcription/medusa_all_500_20260702 \
    --pred finetune_<TS>=./data/processed/transcription/finetune_<TS>_full_corpus \
    --run-name <descriptive_run_name>
```

The sampler's default `SAMPLE_EXCLUDES` list points at both
`full_annotated/` and this folder, so training-batch sampling
automatically excludes these stems — a validation line cannot be
picked into a training batch by construction.

## Invariant

`stems(full_annotated) ∩ stems(validation) = ∅` — verified at merge
time (2026-07-07). Any future annotation batch adds to
`full_annotated/`; the sampler's exclusion rule preserves the
invariant.
