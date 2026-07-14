# OCC_HTR — project spec & running state

> **Purpose of this file.** Persistent state so a fresh agent (after chat
> compaction, or a brand-new session) can pick up where we left off
> without re-asking basics. Instruct the agent to read this first before
> doing anything substantive; then instruct it to keep this file up to
> date with any new decisions, results, or infrastructure changes.
>
> **When to update this file.** Any of:
> - A new experiment ran and produced a metric worth tracking.
> - A convention changed (folder, script naming, logging pattern).
> - A new dataset / model / VM path came into scope.
> - A prior "pending" item is now done — move it to "results" and add
>   the metric.
> - A prior assumption turned out wrong — correct it in place.
>
> Keep it factual. Don't editorialise. Prefer bullet points + short
> paragraphs over prose. Refer out to code / logs when the source of
> truth lives there.

---

## 1. Thesis context

- Manuscript: **AlbucE**, medieval medical text in Old Occitan.
- Task family: OCR / HTR — transcribe segmented line images to text.
- Author: Camila Bermudez Valderrama, LMU Statistics & Data Science MSc,
  supervisor thesis SS2026.
- Deliverable: quantitative comparison of OCR/HTR approaches on this
  specific corpus, with the **permanent 300-line validation set** as the
  common yardstick for every model.

## 2. Repo layout

```
OCC_HTR/
├── data/
│   ├── raw/                              # original manuscript + external corpora
│   │   ├── original_manuscript/
│   │   ├── COMETA_medieval_corpus/
│   │   └── medical_texts/
│   └── processed/
│       ├── extracted_lines/              # raw line PNGs from segmentation
│       ├── filtered_images/              # post ink-bleed filter + double-col corrections
│       │   └── 20260618_160948/original/kept/   # canonical filtered pool
│       ├── annotated_samples/OCR/
│       │   ├── full_annotated/           # 600 hand-verified line pairs (training pool)
│       │   └── validation/               # 300 hand-verified pairs (PERMANENT held-out)
│       ├── transcription/                # per-model prediction folders
│       ├── synthetic_seeds/              # categorized-seed JSONs
│       ├── synthetic_samples/
│       │   ├── augmented_images/aug_20260613_220436/   # kraken aug pool (266k PNGs)
│       │   └── img_labels/labels_20260613_220436/labels.json
│       └── ...
├── src/                                  # business logic; installable package
│   └── ocr/
│       ├── transcribe_img.py             # kraken/catmus inference
│       ├── evaluate_ocr.py               # CER/WER via rapidfuzz
│       ├── finetune.py                   # kraken fine-tune (ketos)
│       ├── medusa_transcribe.py          # Medusa 0.2 Line 9B VLM inference
│       ├── clean_medusa_output.py        # strip chat-template artefacts
│       ├── trocr_finetune.py             # Swin+BERT VisionEncoderDecoderModel training
│       ├── trocr_transcribe.py           # ...and inference
│       └── dictionary_evaluation.py
├── scripts/                              # thin argparse wrappers over src/
│   ├── ocr/run_<same_name>.py
│   └── data_preprocessing/, data_augmentation/, ...
├── frontend/                             # FastAPI + static HTML/JS viewer (§7.4)
│   ├── app.py                            # FastAPI app: /api/pages, /api/pages/{k}, ...
│   ├── manuscript_data.py                # ManuscriptRepo: reads seg JSONs, transcriptions, aligned txt
│   ├── config.py                         # VIEWER_* env-driven paths
│   └── static/                           # index.html + app.js + style.css
├── notebooks/                            # exploratory notebooks (not part of package)
├── models/ocr/                           # checkpoints
│   ├── catmus-medieval.mlmodel           # kraken base model
│   └── finetuned/<run_name>/             # kraken + TrOCR run dirs
├── logs/<task_name>/<run_name>_<task>.log
├── makefile                              # canonical entry point for every stage
├── pyproject.toml                        # single source of truth for deps (uv + setuptools)
├── uv.lock
└── spec.md                               # THIS FILE
```

## 3. Coding conventions

### 3.1 `src/` + `scripts/` split

Every processing step has two files, both named identically:

- **`src/<domain>/<step>.py`** — business logic. Exposes a keyword-arg
  entry point named after the file (`transcribe_image`,
  `run_medusa_transcribe`, `finetune_trocr`, ...). Has module-level
  docstring. Owns logging + config dump. Returns a stats dict.
- **`scripts/<domain>/run_<step>.py`** — thin argparse wrapper. Loads
  `.env` via `python-dotenv`, resolves paths against `PROJECT_ROOT`,
  builds Path defaults, calls the src entry point once, exits.

Reference implementations to mirror when adding new steps:
[src/ocr/medusa_transcribe.py](src/ocr/medusa_transcribe.py) +
[scripts/ocr/run_medusa_transcribe.py](scripts/ocr/run_medusa_transcribe.py).

### 3.2 Logging

Every `src/` entry point:

1. Calls `setup_simple_logging(logs_dir, task_name, run_name)` that
   attaches a `FileHandler` writing to
   `logs/<task_name>/<run_name>_<task_name>.log` **and** a `StreamHandler`.
2. If `log_config=True` (default), dumps a JSON config block near the
   top of the log containing at minimum: `run_name`, short `git_commit`,
   ISO timestamp, all input paths, all knobs, `PROJECT_ROOT` env, and
   the resolved device.
3. **Mirrors aggregate results (metrics tables, summaries) into the
   log** so `logs/<task>/*.log` alone is enough to recover a run's
   numbers — not just the artefact folder. This is a hard rule from
   past feedback; the artefact folders sometimes get moved or deleted.

### 3.3 makefile as canonical entry point

Never run pipeline steps by hand-invoking the CLI unless you're
debugging. `make <target>` is the source of truth for how each step
should be called. Variables at the top of the makefile (`FINETUNE_*`,
`TROCR_*`, `SAMPLE_*`, etc.) hold every path/knob a caller might want
to override.

Add new targets to `.PHONY` and mirror the existing style: a comment
block with example invocations, then the target body using
`$(if $(VAR),--flag $(VAR))` guards for optional flags.

### 3.4 Packaging

- `uv` + setuptools. `pyproject.toml` is the source of truth for deps
  (do **not** edit `requirements.txt`; it's regenerated from
  `pyproject.toml`).
- `src` is the installable top-level package (`from src.ocr import ...`).
- Run commands via `PROJECT_ROOT=. uv run python scripts/<domain>/run_<step>.py ...`
  or, equivalently, `make <target>`.

### 3.5 Explanation style for new tooling

When introducing an unfamiliar library / config knob, give a 1-2
sentence "what is X / why we need it" before dropping it into config —
past feedback preference.

### 3.6 Linter / formatter

Pre-commit hooks: trailing-whitespace, end-of-file-fixer, ruff,
ruff-format. Prefer fixing the code over adding `# noqa`. Hooks
auto-fix ruff-format issues; if a commit fails re-stage and retry.

## 4. Data pipeline (high-level flow)

```
raw manuscript images
  │
  ▼ YOLO layout detection
segmentation JSONs
  │
  ▼ crop_segments
per-page line PNGs (extracted_lines/)
  │
  ▼ binarize + ink-bleed filter + double-column corrections
filtered kept PNGs (filtered_images/.../original/kept/)
  │
  ▼ kraken/catmus baseline transcription  (=CATMUS BASELINE, the OCR seed)
transcription/ocr_kept_20260622_120413/  (per-page .txt)
  │
  ▼ sample_annotation_batch (pre-fills .gt.txt with catmus seed)
tests/ocr/real_val_sample_<TS>/  (annotator edits in place)
  │
  ▼ hand-annotation → merge to permanent pool
data/processed/annotated_samples/OCR/full_annotated/
data/processed/annotated_samples/OCR/validation/
```

## 5. Datasets

### 5.1 Real annotated pool — `data/processed/annotated_samples/OCR/full_annotated/`

- **600 verified `<stem>.png + <stem>.gt.txt` pairs**, built up over 6
  annotation batches (see `_PROVENANCE.md` inside).
- Convention: `.gt.txt` are **NORMALISED** — plain `s` not `ſ`, plain
  `r` not `ꝛ`, `et` not `⁊`. Same across the validation set.
- Sourced from the **filtered** kept folder (not raw extraction), so
  annotators see the same crops the OCR pipeline runs on.
- Used as **training pool** for both kraken and TrOCR fine-tunes.

### 5.2 Permanent validation set — `data/processed/annotated_samples/OCR/validation/`

- **300 verified pairs** (299 non-empty; 1 intentionally-empty line
  `43_f_038v_039_line_22.gt.txt`). Held-out benchmark used for every
  model comparison in the thesis.
- Sampled with `seed=100` across 70 pages, excluding all 600 stems in
  the training pool.
- **Invariant** enforced by the sampler:
  `stems(full_annotated) ∩ stems(validation) = ∅`. Any new training
  batch pulls from the makefile-level default exclude list, which
  already includes `validation/`, so a val line cannot be
  accidentally sampled into training. Sampler:
  [src/data_preprocessing/sample_annotation_batch.py](src/data_preprocessing/sample_annotation_batch.py).
- The evaluation script auto-skips empty gt rows, so effective val
  size for OCR metrics = **299**.

### 5.3 Synthetic augmented pools (disambiguated)

Multiple augmented pools exist on disk. **They are not interchangeable**
— the source text corpus and composition differ. Every training run
records which one it used (all `finetune_*` and `trocr_*` config logs
already do).

**Composition breakdown for each pool** (annotated re-renders =
synthetic renders whose source text came from a real annotated line's
`.gt.txt`; external corpus = renders of text from an unrelated corpus):

| Pool folder | Total | Annotated re-renders | External corpus | Runs using it |
|---|---|---|---|---|
| `aug_20260613_220436/` | 266,478 | 0 (0%) | 266,478 COMETA | kraken `finetune_20260614_133655`; **TrOCR** `trocr_20260710_142341` (Swin+BERT + aug, DONE); **TrOCR** `trocr_20260712_080656` (pretrained TrOCR-base, cancelled). Set as makefile default via `AUGMENTED_RUN_PATH`. |
| `aug_20260629_235051/` | ? | 0 | COMETA | kraken `finetune_20260629_235819` (400 real). |
| `aug_20260701_232640/` | ? | 0 | COMETA | kraken `finetune_20260701_233056` (500 real); `finetune_20260705_070741` (600 real). |
| `aug_merged_anno_medical_20260706/` | 3,000 | 2,000 (400 stems × 5) | 1,000 medical | kraken `finetune_20260706_151856` (600 real + medical). **Note: only 400 of 600 annotated lines re-rendered; imbalance flagged for TrOCR by later work.** |
| **`aug_20260712_124729/`** (2026-07-12) | 3,000 | 3,000 (**all 600 stems × 5**) | 0 | Base for both v2 pools below. Regenerated from all 600 annotated lines via the standard `seeds_from_real → medieval_text_generation → augmentation_techniques` pipeline. |
| **`aug_20260712_v2_matched_cometa/`** (Dataset A'') | 4,000 | 3,000 (600 stems × 5, all annotated) | 1,000 COMETA (seed=42 sample from `aug_20260613_220436`) | **TrOCR** `trocr_20260712_123001` (ViT+RoBERTa + Dataset A'', in progress on VM). Canonical baseline for the 2×3 grid. |
| **`aug_20260712_v2_medical/`** (Dataset B'') | 4,000 | 3,000 (600 stems × 5, all annotated) | 1,000 medical (extracted from `aug_merged_anno_medical_20260706`) | Reserved for TrOCR Runs 2 (ViT+RoBERTa) and 5 (Swin+BERT). Directly comparable to Dataset A''. |

Common properties (apply to every pool):
- Filenames follow `<src_stem>_aug<NN>.png` — annotated re-renders
  additionally carry a `.gt_l<NN>` render-index suffix so the on-disk
  name is `<annotated_stem>.gt_l<NN>_aug<NN>.png`.
- The TrOCR loader's regex (§11 stem-collision fix) strips BOTH
  suffixes so an annotated re-render collapses onto the same source
  stem as its real image — no train/val leak. Kraken's regex is still
  greedy; port the fix when that pipeline is next touched.
- Source is rendered text image-augmented ×N via
  `augmentation_techniques.py`. Labels normalised via `correct_labels.py`
  (plain `s`/`r`/`et`).
- **Not augmentations of the real photos** — these are augmentations
  of synthetic renders. Real photos live in
  `data/processed/annotated_samples/OCR/full_annotated/`.
- Kraken can chew through the full 266k on CPU in a reasonable time.
  TrOCR subsamples to `TROCR_MAX_AUG_SAMPLES=5000` by default; the v2
  pools are already sized at 4000 so no subsampling occurs.

**Rule of thumb for choosing a pool for a new training run:**

- Comparing across the 2×3 TrOCR grid — use `aug_20260712_v2_matched_cometa`
  (Dataset A'') or `aug_20260712_v2_medical` (Dataset B''). These pools
  are symmetric: same 3000 annotated re-renders (all 600 lines × 5),
  differ only in the 1000 external-corpus renders. Isolates the
  "COMETA vs medical corpus source" effect.
- Legacy kraken comparisons — use whichever pool the target row of §6.1
  used; noted in that row.
- Fresh baseline for a new experiment — regenerate a new pool via the
  full `seeds_from_real → medieval_text_generation → augmentation_techniques`
  pipeline. The regenerated pool `aug_20260712_124729` is the reference
  for how a clean "all 600 lines uniformly augmented" pool looks.

### 5.4 Corpora

- `data/raw/COMETA_medieval_corpus/` — general medieval Occitan/Catalan
  text used as synthesis seed for kraken.
- `data/raw/medical_texts/` — medical corpus, 12,012 categorized
  entries (`categorize_20260625_143327/medical_texts_categorized.json`).
  **Was used** in the merged synthetic pool
  `aug_merged_anno_medical_20260706` that trained
  `finetune_20260706_151856` (see §6 kraken catalog).

## 6. Models & results (as of 2026-07-10)

All char/word accuracies are **corpus-level** (Levenshtein distance
via `rapidfuzz`, aggregated over all val lines).

### 6.1 Permanent 300-val benchmark (canonical numbers)

The eval every model is compared against for thesis reporting. Two
eval runs live on disk:
- `tests/ocr/evaluations/seven_way_vs_validation_300/` — the 7-way for
  the pre-2×3-grid catmus/medusa/kraken/legacy-TrOCR comparison.
- `tests/ocr/evaluations/five_trocr_vs_validation_300/` — the 5 grid
  TrOCR runs (built on the VM 2026-07-13, pulled to laptop).

299 lines scored in each (1 gt intentionally empty).

**Corpus-level metrics** (sum of edits / sum of reference chars — one
number over the whole val set; sensitive to a few very bad lines):

| Model | CER | char_acc | WER | word_acc |
|---|---|---|---|---|
| **kraken 600 real** (`finetune_20260705_070741`) | **0.0380** | **0.9620** | 0.2144 | 0.7856 |
| catmus baseline | 0.0387 | 0.9613 | **0.1434** | **0.8566** |
| kraken 500 real (`finetune_20260701_233056`) | 0.0390 | 0.9610 | 0.2188 | 0.7812 |
| kraken 600 real + medical (`finetune_20260706_151856`) | 0.0407 | 0.9593 | 0.2275 | 0.7725 |
| kraken 400 real (`finetune_20260629_235819`) | 0.0420 | 0.9580 | 0.2358 | 0.7642 |
| Medusa 0.2 Line 9B (cleaned v2) | 0.0490 | 0.9510 | 0.3106 | 0.6894 |
| **TrOCR ViT+RoBERTa + medical** (Run 2, `trocr_20260712_150413`) | 0.0557 | 0.9443 | 0.2640 | 0.7360 |
| TrOCR ViT+RoBERTa real-only (Run 3, `trocr_20260713_065604`) | 0.0629 | 0.9371 | 0.2829 | 0.7171 |
| TrOCR ViT+RoBERTa + COMETA (Run 1, `trocr_20260712_123001`) | 0.0668 | 0.9332 | 0.2859 | 0.7141 |
| TrOCR Swin+BERT + medical (Run 5, `trocr_20260713_073113`) | 0.7477 | 0.2523 | 1.0350 | −0.0350 |
| TrOCR Swin+BERT + COMETA (Run 4, `trocr_20260713_071550`) | 0.7760 | 0.2240 | 1.2552 | −0.2552 |
| TrOCR Swin+BERT + aug (legacy `trocr_20260710_142341`) | 0.7101 | 0.2899 | 0.9611 | 0.0389 |

**Per-line median metrics** (median over the 299 lines — describes the
"typical" line rather than the aggregate, robust to a few catastrophic
lines):

| Model | median CER | median char_acc | median WER | median word_acc |
|---|---|---|---|---|
| catmus baseline | 0.0278 | 0.9722 | **0.1250** | **0.8750** |
| Medusa 0.2 Line 9B (cleaned v2) | 0.0435 | 0.9565 | 0.2857 | 0.7143 |
| kraken 400 real (`finetune_20260629_235819`) | 0.0286 | 0.9714 | 0.2000 | 0.8000 |
| kraken 500 real (`finetune_20260701_233056`) | 0.0278 | 0.9722 | 0.1667 | 0.8333 |
| **kraken 600 real** (`finetune_20260705_070741`) | 0.0278 | 0.9722 | 0.1667 | 0.8333 |
| kraken 600 real + medical (`finetune_20260706_151856`) | 0.0278 | 0.9722 | 0.1667 | 0.8333 |
| TrOCR Swin+BERT + aug (`trocr_20260710_142341`) | 0.7209 | 0.2791 | 1.0000 | 0.0000 |

**How to read the medians vs corpus numbers:** every kraken run from
500 lines upward matches catmus's *typical* line (0.0278 CER, 0.9722
char_acc) — the median has hit a floor. The corpus-level char_acc
differences (0.9613 vs 0.9620 etc.) live entirely in the tail: a small
number of hard lines drive the aggregate spread. For thesis reporting,
cite both — the corpus number is what a naive "how good is this model"
question asks, and the median tells you how consistent it is
line-to-line.

**Per-line distribution — CER** (mean / std / percentiles across the
299 scored lines; source of truth is the eval's per-line CSV in
`tests/ocr/evaluations/six_way_vs_validation_300/`):

| Model | corpus | line mean | line std | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| catmus_baseline | 0.0387 | 0.0395 | 0.0459 | 0.0000 | 0.0000 | 0.0000 | 0.0278 | 0.0571 | 0.1000 | 0.2750 |
| medusa_cleaned | 0.0490 | 0.0492 | 0.0438 | 0.0000 | 0.0000 | 0.0250 | 0.0435 | 0.0750 | 0.1031 | 0.2308 |
| kraken_400_real | 0.0420 | 0.0420 | 0.0416 | 0.0000 | 0.0000 | 0.0000 | 0.0286 | 0.0597 | 0.0915 | 0.2703 |
| kraken_500_real | 0.0390 | 0.0391 | 0.0395 | 0.0000 | 0.0000 | 0.0000 | 0.0278 | 0.0571 | 0.0882 | 0.2286 |
| **kraken_600_real** | **0.0380** | **0.0381** | **0.0388** | 0.0000 | 0.0000 | 0.0000 | 0.0278 | 0.0571 | **0.0860** | 0.2286 |
| kraken_600_real_medical | 0.0407 | 0.0408 | 0.0415 | 0.0000 | 0.0000 | 0.0000 | 0.0278 | 0.0588 | 0.0872 | 0.2571 |
| trocr_swin_bert_aug | 0.7101 | 0.7146 | 0.1158 | 0.1579 | 0.5936 | 0.6579 | 0.7209 | 0.7778 | 0.8211 | 1.5556 |

**Per-line distribution — WER:**

| Model | corpus | line mean | line std | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| **catmus_baseline** | **0.1434** | **0.1459** | **0.1567** | 0.0000 | 0.0000 | 0.0000 | **0.1250** | **0.2000** | **0.3750** | **1.0000** |
| medusa_cleaned | 0.3106 | 0.3253 | 0.2783 | 0.0000 | 0.0000 | 0.1250 | 0.2857 | 0.5000 | 0.7143 | 1.4000 |
| kraken_400_real | 0.2358 | 0.2444 | 0.2365 | 0.0000 | 0.0000 | 0.0000 | 0.2000 | 0.3750 | 0.5714 | 1.4000 |
| kraken_500_real | 0.2188 | 0.2290 | 0.2392 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.3750 | 0.5143 | 1.4000 |
| kraken_600_real | 0.2144 | 0.2245 | 0.2382 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.3333 | 0.5000 | 1.4000 |
| kraken_600_real_medical | 0.2275 | 0.2369 | 0.2396 | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.3542 | 0.5714 | 1.4000 |
| trocr_swin_bert_aug | 0.9611 | 0.9800 | 0.1938 | 0.3750 | 0.7722 | 0.8750 | 1.0000 | 1.0000 | 1.1667 | 2.0000 |

**Reading the distributions:**

- **p25 = 0 for CER** across catmus + every kraken run means at least
  25% of lines are transcribed with zero character errors. Medusa's
  p25 = 0.025 — Medusa never quite hits perfect char accuracy on ~75%
  of lines, hinting at a systematic ~1-char offset (normalisation /
  spacing).
- **kraken 400 has p25 = 0 for WER but a mean of 0.244** — the WER
  distribution is bimodal-ish: a good chunk of perfect lines dragged
  down by a long tail of very bad ones (WER up to 1.4).
- **WER max = 1.4 for every non-catmus model** while catmus caps at
  1.0. Every fine-tune has at least one line where the model
  *over-generated* words, driving edit distance above the reference
  length. Matches the `--resize union` codec-widening hypothesis: the
  fine-tune's expanded codec sometimes produces spurious tokens.
- **Distributions are heavily right-skewed** (mean > median for CER).
  Standard "mean ± std" would mislead — cite median + IQR
  (p25 → p75) alongside the corpus number.

**Headline signals:**

- **Medical corpus made the model slightly WORSE**: 0.9593 vs 0.9620.
  This is real data that argues against including the medical corpus
  in the augmentation mix. Possible interpretations: (a) the medical
  vocabulary and rendering are too different from the AlbucE hand and
  the model's capacity gets spent on non-transferable patterns; (b)
  the merged pool traded coverage of AlbucE-typical n-grams for
  broader but less useful diversity.
- **Kraken fine-tunes beat catmus on char_acc but LOSE on WER**.
  Catmus WER is 0.1434 vs kraken best 0.2144. The fine-tunes get
  glyph shapes closer but produce more off-by-one word forms — likely
  a codec/vocabulary side-effect from ``--resize union``.
- **Real-data scale ladder**: 400 → 500 → 600 gives 0.9580 → 0.9610
  → 0.9620. Marginal improvements per +100 real lines.

### 6.2 Historical / biased numbers (kept for provenance)

The 500-pool numbers below were computed BEFORE the permanent val was
carved out — retained for continuity but should not be cited as
generalization scores (self-seeding + train-set overlap bias).

| Model | Train data | Val set | char_acc | Notes |
|---|---|---|---|---|
| catmus-medieval baseline | pretrained only | 500-pool (biased) | 0.9594 | catmus pre-filled the gt.txt seeds |
| Medusa 0.2 Line 9B (raw) | pretrained VLM | 500-pool | 0.8422 | chat-template artefacts |
| Medusa 0.2 Line 9B (cleaned v2) | pretrained VLM | 500-pool | 0.9543 | cleaner strips first-non-noise line |
| kraken `finetune_20260629_235819` | catmus + 400 real | batch-5 (100 unseen) | 0.9624 | prior "fair" generalization test |

### 6.3 TrOCR track — 2×3 grid plan

Two architectures (Swin+BERT from-scratch, ViT+RoBERTa pretrained
`microsoft/trocr-base-handwritten`) × three data conditions:

- **Dataset C** = 600 real only, no aug.
- **Dataset A''** = 600 real + `aug_20260712_v2_matched_cometa` (§5.3)
  = 600 real + 3000 annotated re-renders + 1000 COMETA renders.
- **Dataset B''** = 600 real + `aug_20260712_v2_medical` (§5.3)
  = 600 real + 3000 annotated re-renders + 1000 medical renders.

A'' and B'' differ **only in the 1000 external-corpus slot**, so the
"COMETA vs medical corpus" comparison is clean. The annotated
re-renders (3000 PNGs, 600 stems × 5) are byte-identical between the
two.

**Grid populated with 300-val results (as of 2026-07-13):**

| Architecture | Dataset C (real-only) | Dataset A'' (matched COMETA) | Dataset B'' (medical) |
|---|---|---|---|
| **Swin+BERT from-scratch** | 0.2411 (legacy `_125139`, val-fold) | **0.2240** (Run 4, `_071550`) | **0.2523** (Run 5, `_073113`) |
| **ViT+RoBERTa pretrained** | **0.9371** (Run 3, `_065604`) | **0.9332** (Run 1, `_123001`) | **0.9443** (Run 2, `_150413`) |

All ViT+RoBERTa cells scored against the permanent 300-val via
`run_trocr_transcribe` → `run_evaluate_ocr`; the Swin+BERT cells
likewise. Legacy `_125139` still shows val-fold (would need re-transcribe
against 300-val to be strictly comparable — pending).

**Run execution log** (all on `instance-20260712-110217`, us-west4-c,
L4 GPU, batch_size=32 training + batch_size=16 inference):

| # | Model | Data | Run name | Trained val-fold char_acc | 300-val char_acc | 300-val word_acc |
|---|---|---|---|---|---|---|
| 1 | ViT+RoBERTa pretrained | Dataset A'' | `trocr_20260712_123001` | 0.9643 | 0.9332 | 0.7141 |
| 2 | ViT+RoBERTa pretrained | Dataset B'' | `trocr_20260712_150413` | 0.9654 | **0.9443** | **0.7360** |
| 3 | ViT+RoBERTa pretrained | Dataset C | `trocr_20260713_065604` | 0.9347 | 0.9371 | 0.7171 |
| 4 | Swin+BERT from-scratch | Dataset A'' | `trocr_20260713_071550` | 0.2205 | 0.2240 | −0.2552 |
| 5 | Swin+BERT from-scratch | Dataset B'' | `trocr_20260713_073113` | 0.2395 | 0.2523 | −0.0350 |

Canonical five-way eval CSV+MD:
`tests/ocr/evaluations/five_trocr_vs_validation_300/` (built on the VM,
pulled to laptop after run 5 finished).

Total wall clock: ~5.5h (Run 1 ~1h24m + Run 2 ~1h25m + Run 3 ~12min
early-stopped + Run 4 ~12min early-stopped + Run 5 ~9min early-stopped).
GPU cost: ~$5.

**Key readings from the populated grid:**

- **Pretrained cross-attention closes ~74pp of the char_acc gap** vs.
  the from-scratch build. Same data, only the pretrained TrOCR
  checkpoint changes, jumps 0.22 → 0.94. Clean single-variable
  ablation → strong publishable finding.
- **Medical corpus beats COMETA on 300-val for the pretrained arch**
  (+1.1pp char_acc, +2.2pp word_acc). Opposite of the val-fold reading
  (where they were essentially tied) — because the val-fold contains
  synthetic renders that COMETA reproduces more faithfully, while 300
  real manuscript lines match neither corpus. The medical signal only
  emerges on the real benchmark.
- **Real-only pretrained (Run 3) is very close to COMETA-aug (Run 1)**
  (0.9371 vs 0.9332) — augmentation barely helps the pretrained arch
  at this data scale. Medical actually clears +0.7pp above real-only.
- **All Swin+BERT cells cluster at 0.22-0.25 char_acc**; data barely
  moves the needle. Confirms architecture-bound.
- **Word_acc is negative for Swin+BERT** — WER > 1, i.e. predictions
  contain more edit-distance operations than the reference has words.
  Symptom of over-generation from an unaligned decoder.
- **Best TrOCR (0.9443) still trails kraken 600 real (0.9620) and
  catmus (0.9613)** — catmus and kraken remain the champions on this
  corpus, though the TrOCR pretrained is competitive.

#### 6.3.1 Legacy TrOCR runs (pre-2×3-grid, retained for provenance)

These runs pre-date the pool-matching fix (§5.3) and/or the source-stem
regex fix (§11). They are **not** part of the 2×3 grid comparisons but
kept here for historical reference.

| Model | Train data | Aug pool | Val set | char_acc | word_acc | Notes |
|---|---|---|---|---|---|---|
| Swin+BERT real-only, `trocr_20260710_125139` | 480 real | none | 120 real val-fold | 0.2411 | 0.0000 | Also serves as the Dataset C cell in the grid above — no data / no code change from grid-era version. |
| Swin+BERT + aug, `trocr_20260710_142341` | 600 real + 5000 aug subsampled | `aug_20260613_220436` (COMETA-only, 0 annotated re-renders) | val-fold (1128) | 0.3495 | 0.0890 | Pre-fix + un-matched pool; **not** the Dataset A'' baseline. |
| Swin+BERT + aug, `trocr_20260710_142341` | (same) | (same) | **permanent 300-val** | **0.2899** | **0.0389** | Also listed in §6.1. |
| ViT+RoBERTa pretrained, `trocr_20260712_080656` | 600 real + 5000 aug subsampled | `aug_20260613_220436` (COMETA-only, 0 annotated re-renders) | (cancelled) | — | — | Cancelled 2026-07-12 mid-training when the pool-matching problem was found. Superseded by Run 1 above. |

#### 6.3.2 Conclusion from the Swin+BERT-from-scratch line

Rules out. Even with 10× more data than the real-only baseline (5600
vs 480 pairs), char_acc caps at 0.29 on real photos. The cross-attention
layers, being randomly initialised, would need 50-100× more augmented
data to learn image-text alignment at kraken/catmus quality — not
feasible on this hardware. The 2×3 grid completes the Swin+BERT rows
for symmetry (Runs 4-5), not because we expect them to compete.

#### 6.3.3 Why we care about the pretrained TrOCR

`microsoft/trocr-base-handwritten` ships with cross-attention
pre-trained on 34M synthetic + IAM handwriting lines. Skips the
learn-cross-attention-from-scratch problem entirely. Run 1
(in progress) tells us whether that transfer works for this manuscript
family.

### Kraken fine-tune catalog

Every kraken fine-tune this project has produced, with its training
composition and the source of its augmented synthetic pool. The
"canonical" reporting run is the LAST row unless a later run beats it
on the permanent 300-val benchmark (§6 results row).

| Run | Real | Aug pool | Base model | Notes |
|---|---|---|---|---|
| `finetune_20260629_235819` | 400 | `aug_20260629_235051` (COMETA-only) | catmus-medieval | prior canonical; 320 train + 80 val + synth |
| `finetune_20260701_233056` | 500 | `aug_20260701_232640` (COMETA-only) | catmus-medieval | 400 train + 100 val + synth |
| `finetune_20260705_070741` | 600 | `aug_20260701_232640` (COMETA-only) | catmus-medieval | 480 train + 120 val + synth |
| `finetune_20260706_151856` | 600 | `aug_merged_anno_medical_20260706` (annotated + medical corpus) | catmus-medieval | 480 train + 120 val + merged synth; the medical-corpus run |

Full-corpus transcription output on disk:
- `data/processed/transcription/finetune_400_full_corpus/` — from
  `finetune_20260629_235819`.
- Newer runs have per-line predictions against the val set at
  `data/processed/transcription/<run>_on_validation_300/` once the
  `run_transcribe_line_crops` job (§9 workflow) completes.

Pending baseline runs on the permanent 300-val:
- catmus baseline → 300 val (via `ocr_kept_20260622_120413` rglob).
- Medusa (cleaned) → 300 val (via
  `medusa_validation_300_20260710_clean/`).
- All 4 kraken runs above → 300 val (via
  `<run>_on_validation_300/` folders produced by
  `run_transcribe_line_crops`).

Pending experimental runs:
- **TrOCR Swin+BERT with augmentation** — run `trocr_20260710_142341`
  training as of 2026-07-10 afternoon; update this row + move to results
  table when `final_metrics.json` lands under
  `models/ocr/finetuned/trocr_20260710_142341/`.
- **TrOCR ViT+RoBERTa** starting from `microsoft/trocr-base-handwritten`
  — planned next, gives cross-attention a pretrained starting point.

## 7. Infrastructure

### 7.1 Local laptop

- Apple Silicon Mac. Torch is `2.4.1` (pinned by kraken). MPS available.
- Full training + inference for kraken and TrOCR runs here.
- Some torch ops fall back silently to CPU on MPS — always run TrOCR
  training with `PYTORCH_ENABLE_MPS_FALLBACK=1` prefix so unsupported
  ops fall back to CPU compute instead of crashing.

### 7.2 GCP VMs

Two instances in play, in different zones:

#### 7.2.1 Old CPU-only VM — `instance-20260629-174751`, us-central1-a

- No GPU currently attached (as of 2026-07-10).
- **Two homes, matter which one you use:**
  - `/home/jbermudezv_unal_edu_co/` — where SSH logs in by default.
    This is a fresh account; nothing lives here permanently.
  - `/home/jupyter/OCC_HTR/` — where the **actual repo + `.venv`** live.
    This is the user JupyterLab runs as. Every prior Medusa run
    happened here.
- **No GPU currently attached to this instance** (as of 2026-07-10).
  Medusa 9B on CPU is very slow (~30-60s/line) — 300 lines takes
  several hours. If you need a GPU, spin up a fresh instance.
- Repo on the VM is a git clone with a normal `.venv`; use plain
  `python` (not `uv`) inside it.
- To run Medusa on the VM:
  ```bash
  gcloud compute ssh instance-20260629-174751 --zone=us-central1-a
  sudo -u jupyter -i bash
  cd /home/jupyter/OCC_HTR
  git pull
  source .venv/bin/activate
  # then invoke scripts/ocr/run_medusa_transcribe.py under nohup
  ```
- To copy data laptop → VM, the VM's `~/OCC_HTR/data/...` (i.e.
  `/home/jbermudezv_unal_edu_co/OCC_HTR/data/...`) is where scp lands
  by default. Then `sudo cp -r` into `/home/jupyter/OCC_HTR/data/...`
  and `chown -R jupyter:jupyter` before running.

#### 7.2.2 New L4 GPU VM — `instance-20260712-110217`, us-west4-c

Vertex AI Workbench instance provisioned 2026-07-12 for the TrOCR grid
+ Medusa full-corpus transcription.

- Machine: `n2-standard-8` (8 vCPU, 32 GB RAM) + **NVIDIA L4 × 1**
  (24 GB VRAM, driver 580.65.06, CUDA 13.0).
- Python 3.12.13 (newer than the project's 3.11 constraint —
  workaround: use `PYTHONPATH=.` instead of `pip install -e .` so the
  pyproject `requires-python` check doesn't block).
- Torch 2.12.1+cu130 (CUDA-native, no reinstall needed on this
  instance).
- **transformers 5.12.1 pinned** (NOT 5.13.x — its
  `TokenizersBackend.from_pretrained` breaks on TrOCR-base and can't
  be worked around with `use_fast=False`; see §11).
- Same two-home gotcha as 7.2.1: `gcloud compute ssh` lands you as
  `jupyter`, `gcloud compute scp` writes as `jbermudezv_unal_edu_co`.
  Workaround: scp everything to `/tmp/` (world-writable) first, then
  `cp` into `/home/jupyter/OCC_HTR/` inside the shell.
- Repo at `/home/jupyter/OCC_HTR/`. Data uploads landed at:
  - `data/processed/annotated_samples/OCR/full_annotated/` (600) —
    from git clone (allowlisted in .gitignore).
  - `data/processed/annotated_samples/OCR/validation/` (300) —
    same, from git clone.
  - `data/processed/synthetic_samples/augmented_images/aug_20260712_v2_matched_cometa/` — via tar+scp.
  - `data/processed/synthetic_samples/augmented_images/aug_20260712_v2_medical/` — via tar+scp.
  - `data/processed/filtered_images/20260618_160948/original/kept/` — via tar+scp (500 MB) for the Medusa full-corpus run.
- **Runs on this VM so far:**
  - 5 TrOCR grid runs (§6.3), all trained + 300-val-transcribed
    on-VM. Only the eval CSV/MD was pulled to laptop after (~50 KB).
  - **Medusa full-corpus transcription DONE** —
    `medusa_full_corpus_l4_20260713_095002`. Started 09:50, finished
    15:56 on 2026-07-13. 6.1h wall clock at 0.62 lines/s on L4
    bs=2. All **13,677 lines transcribed, 0 skipped** (after the
    AppleDouble cleanup; see §11). Raw output on VM at
    `data/processed/transcription/medusa_full_corpus_l4_20260713_095002/`;
    cleaned (chat-template artefacts stripped) at
    `data/processed/transcription/medusa_full_corpus_l4_20260713_095002_clean/`.
    Cleaned folder pulled to laptop 2026-07-13 evening — ready for
    the frontend viewer via
    `VIEWER_MODEL_TRANSCRIPTION=./data/processed/transcription/medusa_full_corpus_l4_20260713_095002_clean make frontend`.
    First-attempt run `medusa_full_corpus_l4_20260713_091817` was
    killed at 5-7% when we discovered macOS AppleDouble sidecars
    were doubling the file count; both its output folder and log
    have been deleted.
- Cost: L4 ~$0.7/h. TrOCR grid ~$5 total. Medusa full-corpus
  ~$4. **Stop the instance when idle**: `gcloud compute instances
  stop instance-20260712-110217 --zone=us-west4-c`.
- Deferred: 6 GB tarball at
  `/tmp/occ_htr_vm_runs.tar` on the VM containing every run's
  `best_model/` + metadata + logs. Pulled to laptop 2026-07-13 via
  scp with SSH keepalive at ~6 MB/s.

### 7.3 Model checkpoints on disk

- `models/ocr/catmus-medieval.mlmodel` — kraken base.
- `models/ocr/finetuned/finetune_20260629_235819/model_best.mlmodel` —
  canonical kraken fine-tune (400 real). **Use this whenever you need
  "the fine-tuned kraken" — do not confuse with newer 20260701+ runs
  which were experiments.**

### 7.4 Manuscript viewer (local web app)

FastAPI + vanilla HTML/JS/SVG frontend for exploring the corpus against
model output. Two tabs, both driven off the same page-payload fetch:

- **Tab 1 — transcription viewer.** Original manuscript page on the
  left with clickable segmented-line polygons overlaid as SVG; model
  transcription on the right, one row per line. Clicking either side
  highlights the counterpart. Copy / Download `.txt` buttons pull the
  model transcription for the current page.
- **Tab 2 — 3-way alignment.** Same manuscript image + polygons on the
  left; middle column is the scholarly transcription; right column is
  the model transcription. Clicking a polygon highlights **both** text
  columns so discrepancies pop side-by-side.

Both panes have a zoom toolbar (`−` `+` `⌂` reset), `Cmd`/`Ctrl` +
scroll to zoom under the cursor, and **click-and-drag to pan** (Google
Maps-style — the cursor is `grab` over empty regions of the page and
turns to `grabbing` mid-drag). A short click without meaningful motion
still fires the polygon's normal click handler, so line selection keeps
working alongside drag.

**Data sources** (all resolvable via `VIEWER_*` env vars — see
[frontend/config.py](frontend/config.py)):

- `VIEWER_RAW_PAGES` — raw JPG folder. Default:
  `data/raw/original_manuscript/reproduction14453_100`.
- `VIEWER_SEGMENTATION` — segmentation JSON folder. Default:
  `data/processed/segmented_images/segmentation_20260618_111517`.
- `VIEWER_MODEL_TRANSCRIPTION` — per-line `.txt` root. Default:
  `data/processed/transcription/finetune_400_full_corpus` (canonical
  kraken fine-tune output). **Swap this to whichever model's
  full-corpus transcription you want to inspect** — no code change:
  ```bash
  VIEWER_MODEL_TRANSCRIPTION=./data/processed/transcription/<new_run> make frontend
  ```
- `VIEWER_SCHOLARLY_TXT` — aligned scholarly txt with
  `========== IMAGE: <page_key>_full ==========` headers and `1: ...`
  1-based line entries. Default:
  `tests/ocr/AlbucE_aligned_20260628_142959.txt`.

**Key conventions the viewer relies on:**

- Raw JPG filenames like `5 - garde - 001.jpg` are normalised to the
  same `page_key` used elsewhere (`05_garde_001`) — leading number
  zero-padded to 2 digits, dots/spaces inside a token → `_`, joined
  with `_`.
- Line indices are 0-based in segmentation JSONs and per-line txts;
  the scholarly txt is 1-based and gets converted on parse.
- A page must have BOTH a raw JPG and a segmentation JSON to appear in
  the dropdown. Missing per-line transcription or scholarly text
  renders as muted `— no transcription —` so pipeline gaps stay
  visible.

**Launch:**

```bash
make frontend                          # → http://127.0.0.1:8000
# Override port / host if needed:
make frontend FRONTEND_PORT=9000
```

The FastAPI reloader watches Python files; edits to `static/*.html`,
`.css`, `.js` are picked up by a browser refresh (no server restart).
- `models/ocr/finetuned/trocr_<TS>/best_model/` — TrOCR run outputs
  (each dir is a self-contained VisionEncoderDecoderModel + processor +
  tokenizer, loadable by `trocr_transcribe.py`).

## 8. Command cheat-sheet

All commands assume `cd` into the project root and are runnable via
`make <target>` unless noted. `PYTHON=uv run python` in the makefile
so `uv` handles the venv.

```bash
# 1) Sample a new annotation batch (100 lines, next unused seed).
make sample_annotation_batch SAMPLE_SEED=<NEXT>

# 2) Kraken/catmus baseline transcription over filtered kept PNGs.
make run_transcription   # uses catmus base model

# 3) Kraken fine-tune (real + optional synthetic mix).
make finetune_ocr FINETUNE_EPOCHS=150 FINETUNE_DEVICE=mps

# 4a) TrOCR fine-tune, pretrained checkpoint (RECOMMENDED — see §6.3).
#     Skips learning cross-attention from scratch.
PYTORCH_ENABLE_MPS_FALLBACK=1 make trocr_finetune \
    TROCR_PRETRAINED_MODEL_ID=microsoft/trocr-base-handwritten

# 4b) TrOCR fine-tune, Swin+BERT from-scratch (ablation only; underperforms).
#     Set TROCR_AUGMENTED_FOLDER= TROCR_LABELS_JSON= for real-only.
PYTORCH_ENABLE_MPS_FALLBACK=1 make trocr_finetune

# 5) TrOCR inference against the permanent val set.
make trocr_transcribe \
    TROCR_MODEL_DIR=./models/ocr/finetuned/trocr_<TS>/best_model \
    TROCR_RUN_NAME=trocr_vs_validation_300

# 6) Medusa cleaning (v2 cleaner: takes first non-noise line).
PROJECT_ROOT=. uv run python scripts/ocr/run_clean_medusa_output.py \
    --input-dir <medusa_raw_dir> \
    --output-dir <medusa_clean_dir>

# 7) Evaluate any prediction folder against the permanent val gt.
PROJECT_ROOT=. uv run python scripts/ocr/run_evaluate_ocr.py \
    --gt-dir ./data/processed/annotated_samples/OCR/validation \
    --pred <name>=<pred_dir> \
    [--pred <name2>=<pred_dir2> ...] \
    --run-name <descriptive>

# 8) Manuscript viewer (FastAPI + static HTML/JS/SVG). See §7.4.
make frontend                          # → http://127.0.0.1:8000
# To point Tab 1 at a different model's full-corpus predictions:
VIEWER_MODEL_TRANSCRIPTION=./data/processed/transcription/<run> make frontend
```

## 9. Convention: how to add a new model to the comparison

1. Add training code as `src/<domain>/<model>.py` + wrapper as
   `scripts/<domain>/run_<model>.py`. Match `medusa_transcribe.py`
   shape.
2. Add a `<MODEL>_*` block of variables at the top of the makefile.
3. Add `<model>_transcribe` (and `<model>_finetune` if trainable) to
   `.PHONY` and to the targets section.
4. Produce a prediction folder in
   `data/processed/transcription/<run_name>/` where each file is
   `<stem>.txt`. This is the ONLY layout the evaluator expects.
5. Run `run_evaluate_ocr.py` against the 300-line val set with a
   `--pred name=path` for the new model.
6. Log the resulting char_acc / word_acc in this file's results table
   (§6) and in the run log itself.

## 10. Open questions / decisions to revisit

- **RESOLVED — medical corpus DID NOT help.**
  `finetune_20260706_151856` (600 real + `aug_merged_anno_medical_20260706`)
  scored 0.9593 char_acc vs 0.9620 for the COMETA-only 600-real run
  (`finetune_20260705_070741`). Small but consistent regression. See
  §6.1. Follow-up: do we understand *why*? Options include vocabulary
  drift or wasted capacity — worth a quick error-mode inspection but
  not another training run.
- **Why do kraken fine-tunes lose ~7pp of WER vs. catmus** despite
  matching or beating on char_acc (§6.1 headline signals)? Likely
  ``--resize union`` widening the codec and letting the model produce
  near-neighbour word forms; worth confirming by comparing per-line
  edits.
- **RESOLVED — TrOCR Swin+BERT from scratch is a bust.**
  All three grid cells (real-only, COMETA aug, medical aug) cluster
  at 0.22-0.25 char_acc on the 300-val. Data barely moves the needle
  — architecture-bound.
- **RESOLVED — pretrained TrOCR (ViT+RoBERTa) closes most of the gap
  but doesn't quite beat kraken/catmus.** Best pretrained cell (Run 2,
  medical aug) reaches 0.9443 char_acc on 300-val vs kraken 600
  real's 0.9620 — still 1.7pp behind. See §6.3 grid + §6.1 canonical
  table.
- **RESOLVED — medical corpus HELPS the pretrained TrOCR** (+1.1pp
  char_acc vs COMETA, +2.2pp word_acc). Opposite of kraken (where
  medical hurt slightly). Interpretation: kraken's small CTC model
  gets distracted by non-target-vocabulary; pretrained TrOCR's larger
  capacity absorbs both corpora cleanly and benefits from the
  additional handwriting-like vocabulary in the medical texts.
  Interesting story for the paper: "medical-corpus augmentation is
  architecture-dependent, only helping models with enough capacity to
  ignore the noise".
- If ensembling the top-3 (catmus + kraken 600 + TrOCR pretrained +
  medical) helps the thesis headline number, worth trying. Defer
  until we're sure the individual numbers are stable.

## 11. What NOT to do (past failure modes)

- Don't create `run_transcribe_lines.py` or similar duplicates when
  `run_transcribe_img.py` already handles the case. **Note:**
  `run_transcribe_line_crops.py` is *not* a duplicate — it's the
  flat-folder mode that the page-based `run_transcribe_img.py` doesn't
  cover (val PNGs don't have per-page segmentation JSONs at inference
  time).
- Don't put processing logic in `scripts/<...>/run_*.py` files. Only
  argparse + path resolution. All logic goes in `src/`.
- Don't reference the wrong fine-tune model — the canonical kraken
  fine-tune for reporting is `finetune_20260629_235819` (400 real).
- Don't source annotation batches from the **raw** extraction folder
  (`extracted_lines/extraction_<TS>/`). Always source from the
  **filtered** kept folder
  (`filtered_images/<TS>/original/kept/`) so annotators see the same
  crops the OCR pipeline uses.
- Don't scp uploads / model files to the VM before checking whether
  they're already there. The `/home/jupyter/OCC_HTR/data/` path
  usually already holds prior batches.
- Don't run pipeline steps with hand-invoked CLIs when a `make` target
  exists — you'll drift out of sync with the canonical params.
- Don't skip pre-commit hooks with `--no-verify`. If a hook fails,
  fix the underlying issue (usually ruff auto-format re-stage +
  retry).
- Don't `git checkout <file>` to unstage a partially-staged tracked
  file — that reverts BOTH staged and unstaged changes and silently
  wipes uncommitted work. To split a mixed diff across two commits,
  use `git add -p` and answer `y/n` per hunk, or `git stash` the parts
  you want to defer.
- **Don't upload the full 266k COMETA pool to a compute VM when the
  training will subsample to ~5000 anyway** — the same
  `random.Random(seed).sample(...)` is deterministic, so pre-subsample
  locally into a ~200MB tar and ship that. Saves 30+ min of upload
  per VM run.
- Don't assume "COMETA aug pool" and "medical aug pool" are
  symmetric. The medical pool
  (`aug_merged_anno_medical_20260706`) contains 2000 annotated
  re-renders (400 lines × 5) + 1000 medical corpus renders; the old
  COMETA pool contains 0 annotated re-renders. If you compare them
  directly you're testing *two* variables at once (re-render
  proportion + external corpus). Use the matched v2 pools
  (§5.3) for clean corpus comparisons.
- When TrOCR loading fails on the VM with
  `Couldn't instantiate the backend tokenizer ... You need to have
  sentencepiece or tiktoken installed`, the *fix* is not sentencepiece
  — that's a red-herring error message from **transformers 5.13**.
  Downgrade to `transformers==5.12.1` (our local pinned version).
  Discovered while setting up the L4 VM 2026-07-12.
- Don't assume `gcloud compute scp` lands where `gcloud compute ssh`
  drops you. On Vertex AI Workbench instances, SSH resolves to the
  `jupyter` user but scp uses OS-Login and lands under
  `/home/<sanitised-email>/`. Prefix the scp target with
  `jupyter@instance:...` to pin the same user, or `sudo find /home -name`
  to locate the file when the mismatch strikes. Same two-home gotcha
  as §7.2, different VM.
- **Don't `tar czf` folders on macOS without `COPYFILE_DISABLE=1`** —
  bsdtar packs AppleDouble metadata sidecars (`._<filename>`) for
  every file. On Linux extraction those sidecars become visible files
  and any script iterating the folder (Medusa's `collect_line_images`,
  for instance) will try to open them as real images and fail. Fix:
  prefix the tar command with `COPYFILE_DISABLE=1` or add
  `export COPYFILE_DISABLE=1` to `.zshrc`. Recovery on the VM after
  the fact: `find <dir> -name '._*' -delete`. Cost of not doing this:
  ~50% wasted GPU time before we noticed (2026-07-13 Medusa
  restart).
- **Don't try to scp 6 GB of models over a flaky home connection in
  one shot** — SSH tunnels drop under NAT idle timeouts and scp
  doesn't resume. Options: (a) `--scp-flag='-o
  ServerAliveInterval=60'` for a good connection; (b) split the
  tarball on the VM (`split -b 500M …`) and pull per-part with a
  skip-if-already-present loop; (c) transfer via GCS bucket
  (`gsutil cp`). Or (d) — best — do the work on the VM and pull only
  the small evaluation output; that's what actually worked here.
- Don't try to use transformers 5.13.x for TrOCR pretrained model
  loading. Rebroadcast of the fix from earlier: pin
  `transformers==5.12.1`.
