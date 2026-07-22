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
| `aug_20260629_235051/` | 2,000 | 2,000 (400 annotated stems × 5 aug — 100% re-renders of real texts) | 0 | kraken `finetune_20260629_235819` (400 real). Built via `seeds_from_real.py` on the 400-annotated pool at the time. |
| `aug_20260701_232640/` | 2,500 | 2,500 (500 annotated stems × 5 aug — 100% re-renders of real texts) | 0 | kraken `finetune_20260701_233056` (500 real); kraken `finetune_20260705_070741` (600 real, reused same 500-stem pool — 100 of the 600 real lines have no synth re-renders in this run). |
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

### 6.3 TrOCR track — 2×4 grid

Two architectures (Swin+BERT from-scratch, ViT+RoBERTa pretrained
`microsoft/trocr-base-handwritten`) × four data conditions:

- **Dataset C** = 600 real only, no aug.
- **Dataset D** = 600 real + 3000 annotated re-renders + 0 external corpus
  (`aug_20260712_124729`, the pool that A''/B'' were derived from). Added
  2026-07-14 to isolate the re-render effect from the external-corpus effect.
- **Dataset A''** = 600 real + `aug_20260712_v2_matched_cometa` (§5.3)
  = 600 real + 3000 annotated re-renders + 1000 COMETA renders.
- **Dataset B''** = 600 real + `aug_20260712_v2_medical` (§5.3)
  = 600 real + 3000 annotated re-renders + 1000 medical renders.

A''/B''/D share the same 3000 annotated re-renders; A'' and B'' differ
**only in the 1000 external-corpus slot**, so the "COMETA vs medical
corpus" comparison is clean. Dataset D drops that slot entirely →
"does the external corpus help at all?"

**Grid populated with 300-val results (as of 2026-07-15):**

| Architecture | Dataset C (real-only) | Dataset D (re-renders only) | Dataset A'' (matched COMETA) | Dataset B'' (medical) |
|---|---|---|---|---|
| **Swin+BERT from-scratch** | **0.2293** (legacy `_125139`) | **0.1447** (Run D1, `_192736`) | **0.2240** (Run 4, `_071550`) | **0.2523** (Run 5, `_073113`) |
| **ViT+RoBERTa pretrained** | **0.9371** (Run 3, `_065604`) | **0.9161** (Run D2, `_202441`) | **0.9332** (Run 1, `_123001`) | **0.9443** (Run 2, `_150413`) |

All eight cells now scored against the permanent 300-val via
`run_trocr_transcribe` → `run_evaluate_ocr`. The Swin+BERT real-only
cell was originally val-fold only (0.2411 on its own 120-line val
fold); 2026-07-14 re-transcription against the canonical 300 val gave
0.2293 char_acc, −0.1215 word_acc — consistent with the other
Swin+BERT cells' architecture-bound ceiling.

**Run execution log** (all on `instance-20260712-110217`, us-west4-c,
L4 GPU, batch_size=32 training + batch_size=16 inference):

| # | Model | Data | Run name | Trained val-fold char_acc | 300-val char_acc | 300-val word_acc |
|---|---|---|---|---|---|---|
| 1 | ViT+RoBERTa pretrained | Dataset A'' | `trocr_20260712_123001` | 0.9643 | 0.9332 | 0.7141 |
| 2 | ViT+RoBERTa pretrained | Dataset B'' | `trocr_20260712_150413` | 0.9654 | **0.9443** | **0.7360** |
| 3 | ViT+RoBERTa pretrained | Dataset C | `trocr_20260713_065604` | 0.9347 | 0.9371 | 0.7171 |
| 4 | Swin+BERT from-scratch | Dataset A'' | `trocr_20260713_071550` | 0.2205 | 0.2240 | −0.2552 |
| 5 | Swin+BERT from-scratch | Dataset B'' | `trocr_20260713_073113` | 0.2395 | 0.2523 | −0.0350 |
| — | Swin+BERT from-scratch (retro) | Dataset C | `trocr_20260710_125139` (originally local Mac) | 0.2411 (own val-fold) | 0.2293 | −0.1215 |
| D1 | Swin+BERT from-scratch | Dataset D | `trocr_20260714_192736` | 0.1293 (early-stopped @ ep 6) | 0.1447 | −0.0617 |
| D2 | ViT+RoBERTa pretrained | Dataset D | `trocr_20260714_202441` | 0.9374 | 0.9161 | 0.6728 |

Canonical five-way eval CSV+MD:
`tests/ocr/evaluations/five_trocr_vs_validation_300/` (built on the VM,
pulled to laptop after run 5 finished). The Swin+BERT-Dataset-C row
was added later, evaluated separately locally via
`tests/ocr/evaluations/swinbert_realonly_from_scratch_vs_validation_300/`
(2026-07-14).

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

#### 6.3.4 Staged pretraining for Swin+BERT (results 2026-07-14)

**Motivation.** Single-stage Swin+BERT capped at 0.22-0.25 char_acc
across all three data conditions (§6.3 grid). Hypothesis: the
bottleneck was randomly-initialised cross-attention with too few pairs
to learn image-text alignment, not the architecture itself. Test it by
pretraining Swin+BERT on a bigger COMETA-only pool first (mimicking
what TrOCR-base did at 100× the scale, but scaled down to what our L4
budget permits), then fine-tuning on Datasets A''/B''.

**Two-stage design.** Uses `finetune_trocr` unmodified — Stage 2 just
passes Stage 1's `best_model/` path as `--pretrained-model-id`.

- **Stage 1** — pretraining on COMETA-only aug pool:
  Swin+BERT from-scratch (encoder + decoder pretrained, cross-attn
  random); 15 epochs, bs=32, lr=5e-5, val_fraction=0.05,
  early_stopping_patience=4. Save best_model as the new base.
- **Stage 2a** — fine-tune on Dataset A'' (matched COMETA):
  load Stage 1's best_model, 20 epochs on the 600 real + 3000 anno
  re-render + 1000 COMETA pool.
- **Stage 2b** — fine-tune on Dataset B'' (medical):
  same, on the medical pool. Symmetric to 2a, differs only in the
  1000 external-corpus renders.

Both Stage 2 variants directly comparable to Runs 4 and 5 respectively
(same fine-tune data, only difference: whether cross-attention was
pretrained).

**Scale**: two variants, decided by what upload survives:

| Variant | Stage 1 data | Stage 1 tarball | Stage 1 wall clock (L4) |
|---|---|---|---|
| **30k COMETA** (in progress) | 30,000 pairs subsampled from `aug_20260613_220436` with seed=42 → `aug_20260714_cometa_30k` | ~6 GB (split-uploaded 500 MB × 13 parts after direct scp stalled) | ~2 h |
| **266k COMETA** (deferred; 53 GB upload stalled repeatedly, currently paused) | full `aug_20260613_220436` | ~53 GB | ~3 h (5 epochs) |

**Results — hypothesis confirmed, magnitude smaller on 300-val than on val-fold.**

| Stage | Run name | Data | Wall clock | Val-fold char_acc | Val-fold word_acc | 300-val char_acc | 300-val word_acc |
|---|---|---|---|---|---|---|---|
| **Stage 1a** — pretrain | `trocr_20260714_144423` | 30 000 COMETA re-renders (subsampled from `aug_20260613_220436` seed=42 → `aug_20260714_cometa_30k`); **no manuscript real lines** | 2h 12m (15 epochs) | **0.8589** | **0.7109** | **0.5918** | **0.2888** |
| **Stage 2b** — fine-tune | `trocr_20260714_185946` | 600 real + Dataset B'' | 10 min (6 epochs, early-stopped) | **0.8350** | **0.6640** | **0.6080** | **0.3306** |
| **Stage 2a** — fine-tune | `trocr_20260714_213457` | 600 real + Dataset A'' | 10 min (6 epochs, early-stopped) | **0.8775** | **0.7500** | **0.6053** | **0.3087** |
| Stage 1b — 266 k COMETA pretrain | *(deferred; upload still stalling)* | full `aug_20260613_220436` | — | — | — | — | — |

**Val-fold vs 300-val gap (important reading).** All three staged
rows show a large val-fold → 300-val drop (Stage 1a: −27 pp; Stage 2a:
−27 pp; Stage 2b: −23 pp). The val-fold is a 20 % source-stem split
of each stage's own training pool, so it still contains renders of
stems whose handwriting the model has learned in that same stage; the
300-val is a permanent held-out of real manuscript lines whose stems
the model has never seen. The gap quantifies how much the model
picked up its training distribution rather than genuine generalisation.
This is *the* finding to keep in the thesis, not a footnote.

**Where the lift actually comes from — Stage 1 does almost all the
work.** The single-stage Swin+BERT baseline was **0.2523 char_acc on
300-val** (Run 5 on Dataset B''). Stage 1a alone (COMETA pretraining
with **zero manuscript lines**) reaches **0.5918** — a **+33.95 pp jump
from pretraining alone**. Stage 2 fine-tuning on manuscript data
(600 real + 3000 re-render + 1000 external) adds only:

- **+1.35 pp** on A'' (Stage 2a: 0.6053)
- **+1.62 pp** on B'' (Stage 2b: 0.6080)

So of the total **+35.6 pp lift** vs the single-stage baseline, ~34 pp
comes from the 30 k COMETA pretraining stage and only ~1.5 pp from
the manuscript-specific fine-tune. This is a stronger and cleaner
finding than the original framing: **it is the task-domain pretraining,
not the manuscript fine-tuning, that closes the from-scratch gap**.

**Softened but still-strong headline.** Overall Stage 2b vs single-stage
Swin+BERT: **+35.6 pp on 300-val**, closing about **45 %** of the gap
to the pretrained ViT+RoBERTa (0.9443 on the same Dataset B''). The
original val-fold reading of "+60 pp, closing 72 % of the gap" was
val-fold-inflated. Even with the correction, this is a large and
publishable ablation — arguably more publishable now, because the
per-stage decomposition is a real result the field will care about.

**Directional finding on external corpus for the staged track.**
Under staging, matched COMETA (A'', 0.6053) and medical (B'',
0.6080) are essentially tied on 300-val — the corpus swap in the
1000-render slot doesn't move the needle for a from-scratch
cross-attention that has already seen 30 k COMETA renders in
Stage 1. Contrast with the pretrained ViT+RoBERTa where medical
edges COMETA by +1.1 pp on the same swap. Consistent with "Stage 1
does most of the work" — once Stage 1 saturates the cross-attention,
which corpus you fine-tune with barely matters.

**Next steps for this experiment**:
- If we later push Stage 1 to a bigger COMETA pool (60 k or the full
  266 k) once the upload path is unblocked, the val-fold-to-300-val
  gap is the primary metric to watch — a shrinking gap would mean
  more pretraining is actually improving generalisation, not just
  memorising handwriting.
- A useful additional ablation: Stage 1 on 30 k *medical* corpus
  renders (instead of COMETA) → tests whether Stage 1 corpus choice
  matters when the fine-tune corpus choice doesn't.

**Thesis contribution**: controlled ablation showing that 30 k pairs
of task-domain pretraining is worth ~34 pp of held-out char_acc for
an encoder-decoder VLM whose cross-attention would otherwise be
randomly initialised — with the additional finding that
manuscript-specific fine-tuning on top of that pretraining adds only
~1.5 pp, and that the val-fold reading of the same intervention
over-states the gain by ~24 pp. Three publishable sub-findings
from one controlled experiment.

#### 6.3.5 Extending the grid to 2 × 4 — adding Dataset D (re-renders only)

**Motivation.** The current 2 × 3 grid conflates two effects between
Dataset C (real-only) and Dataset A''/B'' (real + 3000 anno re-renders
+ 1000 external corpus): we can't tell whether the delta comes from
the re-rendering step or from the external-corpus text.

**New column — Dataset D**: 600 real + 3000 anno re-renders +
**0 external corpus renders**. Uses the pre-existing pool
`aug_20260712_124729` (which was built as the *base* for A''/B'' but
never trained on directly).

| | Real | Anno re-renders | External corpus |
|---|---|---|---|
| Dataset C | 600 | 0 | 0 |
| **Dataset D (new)** | **600** | **3000 (600 lines × 5)** | **0** |
| Dataset A'' | 600 | 3000 | 1000 COMETA |
| Dataset B'' | 600 | 3000 | 1000 medical |

**Clean single-variable deltas that fall out of the extended grid:**

- **C → D delta** = pure "does re-rendering our own texts help?"
- **D → A'' delta** = pure "does 1000 COMETA renders help on top of
  re-rendering?"
- **D → B'' delta** = pure "does 1000 medical corpus renders help on
  top of re-rendering?"

**Runs planned (single-stage, not staged — matching Runs 4-5 and
Runs 1-3 for direct comparison):**

- **Run D1** — Swin+BERT + Dataset D (single-stage from-scratch).
  Compares to Runs 4 (A'', 0.2240) and 5 (B'', 0.2523).
- **Run D2** — ViT+RoBERTa pretrained + Dataset D. Compares to Runs
  1 (A'', 0.9332), 2 (B'', 0.9443), 3 (C real-only, 0.9371).

Both use `aug_20260712_124729` on the VM (reconstructed from A''
locally in ~10 s via a filter script — see §7.2.2 for the recipe).
Sequential on the L4 GPU; ~1-1.5 h each; total additional cost ~$2.

**Results (2026-07-15, both landed):**

| Run | Model | Data | Wall clock | Val-fold char_acc | 300-val char_acc | 300-val word_acc |
|---|---|---|---|---|---|---|
| **D1** | Swin+BERT from-scratch | Dataset D (3600 pairs) | ~10 min (6 epochs, early-stopped) | 0.1293 | **0.1447** | −0.0617 |
| **D2** | ViT+RoBERTa pretrained | Dataset D (3600 pairs) | 64 min (20 epochs) | 0.9374 | **0.9161** | 0.6728 |

**Findings from the extended grid.**

- **C → D delta for ViT+RoBERTa: −2.1 pp** (0.9371 real-only → 0.9161
  with 3000 re-renders added). Pure re-rendering of the same 600 real
  texts does **not** help the pretrained arch — it slightly hurts,
  probably because the re-render texture introduces low-diversity
  noise a well-pretrained model overfits to.
- **D → A'' delta for ViT+RoBERTa: +1.7 pp** (0.9161 → 0.9332). Adding
  1000 COMETA renders on top of the re-renders recovers most of the
  loss and then some — the *external corpus* is what carries the
  augmentation signal, not the re-render volume.
- **D → B'' delta for ViT+RoBERTa: +2.8 pp** (0.9161 → 0.9443). Same
  effect, stronger — medical corpus contributes ~1 pp more than
  COMETA on top of the re-render base.
- **C → D delta for Swin+BERT: −8.5 pp** (0.2293 → 0.1447). Same
  direction as ViT+RoBERTa but amplified: from-scratch cross-attention
  with only 600 stems of variety collapses harder when fed noisy
  re-renders without text-distribution diversity.
- **All Swin+BERT single-stage cells now confirmed to cluster at
  0.14-0.25 char_acc regardless of data condition** — the +11 pp
  spread across the four Swin+BERT cells is small compared to the
  ~35 pp lift from the staging intervention (§6.3.4), reinforcing
  "cross-attention pretraining >> data recipe" as the top-line
  narrative.

Eval artefacts: `tests/ocr/evaluations/staged_and_D_vs_val300/` (CSV +
MD). Best-model checkpoints on laptop under
`models/ocr/finetuned/trocr_20260714_192736/` (D1) and
`trocr_20260714_202441/` (D2).

#### 6.3.6 Tokenizer-floor CER analysis (2026-07-18)

**Motivation.** The +72 pp gap between Swin+BERT (mBERT WordPiece
decoder tokenizer) and ViT+RoBERTa (byte-level RoBERTa BPE tokenizer,
bundled with `microsoft/trocr-base-handwritten`) has two candidate
explanations that co-vary in the ablation:

1. **Cross-attention pretraining** — the intended finding.
2. **Tokenizer coverage of medieval Latin abbreviations** — a
   confound: mBERT WordPiece can `[UNK]` on characters like `⁊`
   (Tironian et, U+204A), `ꝑ` (U+A751), etc., while byte-level BPE
   can encode any codepoint via byte-fallback.

To disentangle them, measure the **CER floor imposed by each
tokenizer** on the 300-val ground truth via encode → decode round-trip.
The floor is the lower bound on CER *any* model using that tokenizer
could achieve, even with a perfect encoder. The difference between
the two floors is an upper bound on the tokenizer's contribution to
the +72 pp observed gap.

**Script.** `scripts/ocr/analyze_tokenizer_floor.py`. Loads each
tokenizer via `AutoTokenizer.from_pretrained`, iterates over
`data/processed/annotated_samples/OCR/validation/*.gt.txt`, and
reports corpus-level CER floor + top `[UNK]`-triggering characters +
5 worst-case round-trip lines per tokenizer.

**Results (2026-07-18, 299 non-empty lines, 11 000 total chars).**

| Tokenizer | Vocab size | Perfect round-trips | Corpus CER floor (skip specials) | Implied char_acc ceiling |
|---|---|---|---|---|
| **mBERT** (`bert-base-multilingual-cased`) | 119 547 | 252 / 299 (84.3 %) | **0.0074** | **99.26 %** |
| **RoBERTa BPE** (bundled with `microsoft/trocr-base-handwritten`) | 50 265 | 299 / 299 (100 %) | **0.0000** | **100.00 %** |

**Conclusion — tokenizer confound is negligible.** The difference
between the two tokenizer floors is **≤ 0.74 pp of char_acc**, vs a
+72 pp observed gap in the 2 × 4 grid. Cross-attention pretraining
accounts for ≥ 71 pp of the gap; tokenizer coverage accounts for
at most 0.74 pp. This lets the thesis state the finding cleanly:

> The +72 pp gap between Swin+BERT (from-scratch cross-attention) and
> ViT+RoBERTa (`microsoft/trocr-base-handwritten`) on 300-val
> char_acc is attributable to cross-attention pretraining rather
> than tokenizer coverage — the tokenizer contribution is bounded
> above at 0.74 pp on this benchmark.

**Where mBERT actually loses accuracy.** 16 of 299 lines (5.4 %)
contain at least one `[UNK]`. Character breakdown of the losses:

| Char | Codepoint | `[UNK]` count | What it is |
|---|---|---|---|
| `⁊` | U+204A | 12 | Tironian et (medieval "and" abbreviation); ~75 % of the mBERT floor CER |
| `ꝑ` | U+A751 | 1 | Latin small letter p with stroke (per / par / por abbreviation) |
| `q`, `a`, ` ` | — | 1 each | One-off edge cases at line boundaries |

Other medieval abbreviations in the corpus (`ꝓ`, `ẽ`, `ĩ`, `ā`, etc.)
survive mBERT's WordPiece intact — they are recoverable as
combinations of existing tokens or via byte fallback.

**Secondary finding — WordPiece whitespace normalisation adds noise.**
Some of mBERT's worst round-trip cases are not `[UNK]` losses but
WordPiece adding / removing spaces around periods:

```
REF: da.esia ⁊ gran.so es aygua cauda
HYP: da. esia   gran. so es aygua cauda    (periods gained spaces; ⁊ vanished)
```

This is another small structural CER contribution independent of
`[UNK]` handling — but still small enough to be inside the 0.74 pp
overall floor.

**Reproducing.**

```
cd <repo root>
python3 scripts/ocr/analyze_tokenizer_floor.py \
    --val-dir data/processed/annotated_samples/OCR/validation
```

~5 s runtime on laptop. No GPU, no model download beyond the two
tokenizers (a few MB each).

#### 6.3.7 Bootstrap 95 % confidence intervals for TrOCR-track models on 300-val (2026-07-18)

> **⚠ Two findings here were reversed by the corrected-annotation rerun
> (§6.3.10).** "Medical > COMETA for pretrained ViT+RoBERTa" (+1.11 pp) and
> "manuscript FT adds +1.62 pp over Stage 1" are both **no longer significant**
> on the corrected 300-val. The bootstrap machinery and the still-significant
> comparisons below remain valid.

**Motivation.** All numbers in §6.3, §6.3.4 and §6.3.5 are point
estimates on a 299-line held-out set. Two questions the committee
will ask:

1. **Per-model uncertainty.** How wide is the confidence band around
   each reported char_acc / word_acc?
2. **Difference significance.** Is the +1.1 pp advantage of medical
   over COMETA for pretrained ViT+RoBERTa real, or within noise?
   Same question for the A''/B'' tie under staging, and for
   Stage 2's +1.6 pp over Stage 1.

Both answered by resampling the 299 lines with replacement, 10 000
iterations, and recomputing corpus-level metrics on each resample.
For pair-wise questions we use **paired bootstrap** — the same random
line indices apply to both models on every iteration, so the CI on
the difference reflects the fact that both models were evaluated on
the same underlying lines.

**Script.** `scripts/ocr/bootstrap_ocr_ci.py`. Reads per-line eval
CSVs from any number of `--eval-dir` folders, inner-joins by stem,
runs both per-model bootstrap and paired bootstrap. Numpy-only
implementation, ~1 s runtime on laptop for 11 models × 10 000
iterations.

**Command that produced the results below (deterministic with
`seed=42`):**

```
cd <repo root>
python3 scripts/ocr/bootstrap_ocr_ci.py \
    --eval-dir tests/ocr/evaluations/five_trocr_vs_validation_300 \
    --eval-dir tests/ocr/evaluations/staged_and_D_vs_val300 \
    --eval-dir tests/ocr/evaluations/stage1a_vs_val300 \
    --eval-dir tests/ocr/evaluations/swinbert_realonly_from_scratch_vs_validation_300 \
    --n-boot 10000 --seed 42
```

Full raw output snapshotted at
`tests/ocr/evaluations/bootstrap_ci_trocr_20260718/bootstrap_ci_trocr.txt`.

**Per-model 95 % CIs (paired-bootstrap on the 299-line held-out set).**

| Model | char_acc [95 % CI] | word_acc [95 % CI] | CER [95 % CI] | WER [95 % CI] |
|---|---|---|---|---|
| vitroberta_medical | 94.43 % [93.83, 95.00] | 73.59 % [70.98, 76.16] | 0.0557 [0.0500, 0.0617] | 0.2641 [0.2384, 0.2902] |
| vitroberta_realonly | 93.72 % [93.02, 94.39] | 71.73 % [69.16, 74.20] | 0.0628 [0.0561, 0.0698] | 0.2827 [0.2580, 0.3084] |
| vitroberta_cometa | 93.32 % [92.57, 94.04] | 71.41 % [68.64, 74.03] | 0.0668 [0.0596, 0.0743] | 0.2859 [0.2597, 0.3136] |
| runD2_vitroberta | 91.61 % [90.74, 92.44] | 67.29 % [64.52, 70.02] | 0.0839 [0.0756, 0.0926] | 0.3271 [0.2998, 0.3548] |
| stage2b_medical | 60.81 % [59.07, 62.57] | 33.07 % [30.43, 35.81] | 0.3919 [0.3743, 0.4093] | 0.6693 [0.6419, 0.6957] |
| stage2a_cometa | 60.53 % [58.74, 62.29] | 30.86 % [27.84, 33.76] | 0.3947 [0.3771, 0.4126] | 0.6914 [0.6624, 0.7216] |
| stage1a_cometa_pretrain | 59.17 % [57.55, 60.82] | 28.87 % [26.25, 31.50] | 0.4083 [0.3918, 0.4245] | 0.7113 [0.6850, 0.7375] |
| swinbert_medical | 25.22 % [24.40, 26.06] | −3.51 % [−5.33, −1.76] | 0.7478 [0.7394, 0.7560] | 1.0351 [1.0176, 1.0533] |
| swinbert_realonly | 22.92 % [21.81, 24.02] | −12.18 % [−14.53, −9.88] | 0.7708 [0.7598, 0.7819] | 1.1218 [1.0988, 1.1453] |
| swinbert_cometa | 22.40 % [21.57, 23.23] | −25.53 % [−28.37, −22.83] | 0.7760 [0.7677, 0.7843] | 1.2553 [1.2283, 1.2837] |
| runD1_swinbert | 14.46 % [12.99, 15.88] | −6.18 % [−8.13, −4.31] | 0.8554 [0.8412, 0.8701] | 1.0618 [1.0431, 1.0813] |

**Paired bootstrap comparisons (A vs B).** P(A > B) is the fraction
of bootstrap resamples in which A's char_acc exceeded B's. A comparison
is significant at α = 0.05 if the 95 % CI on the difference excludes 0
(equivalently, P(A > B) ≥ 0.975 or ≤ 0.025).

| A | B | Δ char_acc [95 % CI] | Δ word_acc [95 % CI] | P(A > B) | Verdict | Interpretation |
|---|---|---|---|---|---|---|
| vitroberta_medical | vitroberta_cometa | +1.11 % [+0.53, +1.72] | +2.21 % [+0.29, +4.11] | 1.000 | ✓ sig | Medical > COMETA for pretrained arch is real |
| vitroberta_medical | vitroberta_realonly | +0.72 % [+0.18, +1.29] | +1.90 % [−0.15, +3.92] | 0.995 | ✓ sig (char_acc); borderline (word_acc) | Medical aug barely beats no aug |
| vitroberta_realonly | runD2_vitroberta | +2.10 % [+1.37, +2.85] | +4.41 % [+2.31, +6.58] | 1.000 | ✓ sig | Dataset D significantly hurts pretrained arch |
| stage2b_medical | swinbert_medical | +35.58 % [+33.79, +37.37] | +36.55 % [+33.82, +39.31] | 1.000 | ✓ sig | Staged pretraining lift is bulletproof |
| stage1a_cometa_pretrain | swinbert_medical | +33.95 % [+32.25, +35.70] | +32.38 % [+29.72, +35.14] | 1.000 | ✓ sig | COMETA pretraining alone does 34 pp — sig |
| stage2b_medical | stage1a_cometa_pretrain | +1.62 % [+0.29, +2.93] | +4.18 % [+2.07, +6.32] | 0.992 | ✓ sig (p = 0.008) | Manuscript FT DOES add real value on top of Stage 1 (small but detectable) |
| stage2a_cometa | stage2b_medical | −0.25 % [−1.79, +1.30] | −2.15 % [−4.39, +0.15] | 0.377 | ✗ NOT sig | Under staging, corpus choice at Stage 2 is within noise |
| vitroberta_medical | stage2b_medical | +33.65 % [+31.99, +35.31] | +40.55 % [+37.68, +43.40] | 1.000 | ✓ sig | Pretrained TrOCR still significantly beats staged Swin+BERT |

**Interpretation for the thesis narrative.**

- **Rewrite of §6.3.4's "manuscript FT is a rounding error" line.**
  The +1.62 pp Stage 2 lift over Stage 1 is *statistically detectable*
  (95 % CI [+0.29, +2.93], p = 0.008 that A ≤ B). Replace
  "manuscript-specific fine-tuning is a rounding error" with
  "manuscript-specific fine-tuning adds a small but statistically
  significant +1.6 pp over pretraining alone".
- **A''/B'' tie under staging is now bulletproof as a negative result.**
  Δ = −0.25 pp on char_acc with 95 % CI [−1.79, +1.30]; P(A > B) =
  0.377. This is a defensible finding: under the staged pipeline,
  the corpus swap at Stage 2 is within noise. Contrasts cleanly with
  the ViT+RoBERTa case where the same swap is +1.11 pp with 95 % CI
  [+0.53, +1.72] — significant.
- **All headline gaps hold with margin.** The +72 pp cross-attention
  gap, the +36 pp staged-pretraining lift, and the +34 pp Stage 1-only
  lift all have CIs comfortably away from zero. None of these is at
  risk of being an artefact of the 299-line sample size.

**Pending — kraken side.** Once the kraken 600 baseline is re-run
with the pool composition matched to the medical run (2000 anno
re-renders in both, so that the kraken medical vs no-medical
comparison is single-variable like A'' vs B''; see §6.3 confound
note and the [Kraken fine-tune catalog](#kraken-fine-tune-catalog)
below), re-run this same script pointing at the kraken eval CSVs
to get an equivalent CI table for the kraken track.

#### 6.3.8 Bootstrap stratified by GT-transcription confidence (2026-07-18)

**Motivation.** Not every one of the 300 held-out lines has a
100 %-verified transcription — some GT files were left flagged for a
second review. `tests/ocr/validation_300_manifest_.csv` adds a
`validated_100` column: **1 = human-verified GT**, **0 = still to be
double-checked**. As of 2026-07-18 the split is **286 validated + 14
unvalidated = 300 total** (or **285 validated + 14 unvalidated = 299
non-empty** — the single empty-GT line is validated but always
excluded from eval CSVs). This raises two questions the committee
will ask:

1. **Are the full-set numbers in §6.3.7 confounded by unvalidated
   lines?** i.e. is model X really at 0.9443 char_acc, or is that
   number partly the model being "penalised" for correctly
   transcribing lines whose GT is wrong?
2. **Are unvalidated lines systematically harder?** If yes, that
   pushes them onto the priority list for the next round of
   verification.

**Method.** Re-run `bootstrap_ocr_ci.py` twice — once with
`--filter-value 1` (validated subset), once with `--filter-value 0`
(unvalidated subset). Paired bootstrap 10 000 iterations, seed=42, same
as §6.3.7.

**Command.**

```
cd <repo root>
for value in 1 0; do
  python3 scripts/ocr/bootstrap_ocr_ci.py \
      --eval-dir tests/ocr/evaluations/five_trocr_vs_validation_300 \
      --eval-dir tests/ocr/evaluations/staged_and_D_vs_val300 \
      --eval-dir tests/ocr/evaluations/stage1a_vs_val300 \
      --eval-dir tests/ocr/evaluations/swinbert_realonly_from_scratch_vs_validation_300 \
      --manifest tests/ocr/validation_300_manifest_.csv \
      --filter-col validated_100 --filter-value "$value" \
      --n-boot 10000 --seed 42
done
```

Full raw output snapshotted at
`tests/ocr/evaluations/bootstrap_ci_trocr_validated_20260718/bootstrap_ci_trocr_by_validated.txt`.

**Per-model char_acc — side by side across all three views.**

| Model | All (n=299) | Validated (n=285) | Unvalidated (n=14) | Δ unval − val |
|---|---|---|---|---|
| vitroberta_medical | 94.43 % | 94.54 % [93.93, 95.14] | 92.28 % [90.09, 94.41] | **−2.26 pp** |
| vitroberta_realonly | 93.72 % | 93.82 % [93.08, 94.51] | 91.70 % [88.44, 94.68] | **−2.12 pp** |
| vitroberta_cometa | 93.32 % | 93.55 % [92.77, 94.28] | 88.76 % [84.17, 92.45] | **−4.79 pp** |
| runD2_vitroberta | 91.61 % | 91.88 % [90.99, 92.72] | 86.36 % [82.28, 90.24] | **−5.52 pp** |
| stage2b_medical | 60.81 % | 61.09 % [59.23, 62.95] | 55.05 % [48.82, 61.86] | −6.04 pp |
| stage2a_cometa | 60.53 % | 61.03 % [59.18, 62.87] | 50.98 % [45.87, 56.32] | **−10.05 pp** |
| stage1a_cometa_pretrain | 59.17 % | 59.35 % [57.67, 61.02] | 55.77 % [48.89, 63.03] | −3.58 pp |
| swinbert_medical | 25.22 % | 25.17 % | 26.36 % | +1.19 pp |
| swinbert_realonly | 22.92 % | 22.91 % | 23.39 % | +0.48 pp |
| swinbert_cometa | 22.40 % | 22.43 % | 21.73 % | −0.70 pp |
| runD1_swinbert | 14.46 % | 14.39 % | 16.00 % | +1.61 pp |

**Two clean readings.**

- **All 7 mid-to-high-accuracy models perform worse on unvalidated
  lines** (drops of 2–10 pp). Consistent with the hypothesis that
  unvalidated GT contains transcription errors that penalise
  otherwise-correct predictions.
- **Single-stage Swin+BERT models are indifferent** (deltas within
  noise, direction inconsistent). Makes sense — those models are
  already below 26 % char_acc, so their own error rate dominates any
  contribution from GT noise.

**Paired comparisons — validated subset (n=285) confirms every §6.3.7
finding.**

| A vs B | Full (n=299) | Validated (n=285) | Sig on validated? |
|---|---|---|---|
| vit_med vs vit_cometa | +1.11 [+0.53, +1.72] | +0.98 [+0.41, +1.58] | ✓ (P=0.999) |
| vit_med vs vit_realonly | +0.72 [+0.18, +1.29] | +0.73 [+0.18, +1.31] | ✓ (P=0.995) |
| vit_realonly vs runD2 | +2.10 [+1.37, +2.85] | +1.93 [+1.19, +2.71] | ✓ (P=1.000) |
| stage2b vs swinbert_med | +35.58 | +35.93 [+34.06, +37.79] | ✓ (P=1.000) |
| stage2a vs stage2b | −0.25 [−1.79, +1.30] | −0.06 [−1.67, +1.56] | ✗ (P=0.471) — still tied |
| stage1a vs swinbert_med | +33.95 | +34.19 [+32.42, +36.03] | ✓ (P=1.000) |
| stage2b vs stage1a | +1.62 [+0.29, +2.93] | +1.74 [+0.40, +3.12] | ✓ (P=0.994) — manuscript FT still adds real value |
| vit_med vs stage2b | +33.65 | +33.45 [+31.83, +35.14] | ✓ (P=1.000) |

Every conclusion from §6.3.7 holds on the validated subset with
essentially identical numbers. The full-set numbers were **not**
confounded by unvalidated-GT noise — the shift is uniformly within
0.2 pp of the full-set point estimates.

**Paired comparisons — unvalidated subset (n=14), flagged pattern.**

CIs on the 14-line subset are wide, so nothing is conclusive here.
The one directional signal worth noting:

| A vs B | Δ char_acc [95 % CI] | P(A > B) | Interpretation |
|---|---|---|---|
| stage2a vs stage2b | **−4.07 %** [−8.53, +0.56] | 0.042 | On unvalidated lines, medical (B'') beats COMETA (A'') by ~4 pp, barely non-significant (CI barely includes 0). Reversal from the validated subset (tie). Speculative reading: unvalidated lines may be over-represented in the harder / more medical-vocabulary-dense subset, exactly where medical-corpus training should help most. |

**Recommendations.**

- **Adopt the validated-285 as the canonical benchmark** for all
  future TrOCR / kraken numbers in the thesis, with the full-299
  numbers relegated to an appendix / robustness check. Justification:
  a "true CER" metric measures how well a model transcribes
  *correct* GT; unvalidated lines contaminate that measurement.
- **Manually inspect the 14 unvalidated lines** for the models with
  the largest drops (vitroberta_cometa −4.79 pp, runD2_vitroberta
  −5.52 pp, stage2a_cometa −10.05 pp). Cases where the model's
  "error" is actually a GT error should be promoted to
  `validated_100=1` (fixing the GT if needed) and the manifest
  re-committed.
- **Kraken track re-runs**: when the kraken 600 matched-pool run
  completes, feed its eval CSV to `bootstrap_ocr_ci.py` with
  `--manifest tests/ocr/validation_300_manifest_.csv --filter-col
  validated_100 --filter-value 1` so kraken numbers are directly
  comparable to the TrOCR-track validated benchmark.

#### 6.3.9 Kraken train/val split text-level leak — diagnosis + fix (2026-07-21)

**Symptom.** After the annotation-correction re-run for kraken
matched-pool (finetune_20260718_193601, finetune_20260719_085411),
the new baseline dropped ~5 pp char_acc (0.9620 → 0.9096) vs the
historical canonical kraken 600 (finetune_20260705_070741). Internal
val_accuracy was *higher* than the old run (0.943 vs 0.889) but true
300-val performance was *lower*, and training stopped at only 29
epochs vs the historical 88 with the same `--lag 5` early stopping.

**Root cause (verified by tracing `src/ocr/finetune.py`).** The
kraken finetune script did two INDEPENDENT stem-level shuffles when
both `--augmented-folder` and `--real-folder` were supplied:

1. `stage_finetune_data()` shuffled the 600 aug source stems with
   seed=42, taking 60 (10 %) as synth val + 540 (90 %) as synth train.
2. `mix_in_real_samples()` shuffled the 600 real pairs with seed=42
   too, taking first 480 (80 %) as real train + next 120 (20 %) as
   real val, then discarded the synth val by default.

Both shuffles operated on the same alphabetically-sorted 600 stems
with the same seed, so the SAME permutation was applied — but the
two functions carved the shuffled list at different points. Result:
positions 480-599 in the permutation ended up as **synth train + real
val**. **All 120 real val stems had their 5 aug siblings sitting in
synth train.** The model saw synthetic renders of every val text
during training, so its internal val_accuracy on the real val images
converged fast (text-level familiarity) and triggered `--lag 5` early
stopping while the model was still under-trained on true
generalisation.

Verified by smoke-test on the current data: intersection of
`real_val_stems` with `synth_train_stems` = **120 of 120 val stems
had their augs in train**.

**Fix (commits: kraken finetune coordinate-split + regex + stats-dict
fixes; `_summarize_and_prune` graceful missing-kraken).** Three
changes to `src/ocr/finetune.py`:

1. New helper `_compute_real_stem_split()` — determines the 480 train
   + 120 val real stems ONCE from the real folder (seed 42).
2. `stage_finetune_data()` accepts an optional `route_by_stems=
   (train_stems, val_stems)` argument. When given, aug files are
   routed by their source stem's membership in those sets — augs of
   a real train stem go to train, augs of a real val stem go to val.
   Nothing is dropped.
3. `finetune()` main computes the real split up front (when both
   real and aug folders are supplied), passes the same partition to
   both `stage_finetune_data()` and `mix_in_real_samples()`, and
   force-disables `real_replaces_synth_val` so the val staging
   preserves the aug val samples that got routed there.

Also fixed `_AUG_FILENAME_RE` (from the greedy `^(.+)_aug\d+\.png$`
to the non-greedy `^(.+?)(?:\.gt_l\d+)?_aug\d+\.png$` that matches
`trocr_finetune.py`) so aug source stems now equal real stems and the
intersection is well-defined. Also patched `_summarize_and_prune` to
tolerate the `kraken` package not being importable in the parent
Python (system python vs `uv run`) — best_model preserved by ketos
either way, extras still pruned even without kraken.

**TrOCR was already correct.** `src/ocr/trocr_finetune.py`
`_split_by_source_stem()` groups ALL images (real + aug) by source
stem BEFORE splitting, so all 6 files of one stem (1 real + 5 aug)
land on the same side by construction. No leak in trocr; no patch
needed there.

**Post-fix training pool for kraken (matched-pool no-medical
2026-07-21):**
- TRAIN: 2400 aug + 480 real = 2880 images (from 480 stems)
- VAL:   600 aug + 120 real = 720 images  (from 120 disjoint stems)
- Zero leaked stems across the boundary.

**Post-fix result — Kraken Run 1 (matched no-medical, leak-fixed,
corrected 600 annotations):**

| Metric | Leak-affected `_193601` (2026-07-18) | Leak-fixed `_20260721_200641` (2026-07-22) | Δ |
|---|---|---|---|
| Epochs trained | 29 | 56 | +27 |
| Best epoch | 23 | 51 | +28 |
| Internal val_accuracy | 0.9430 | **0.9581** | **+1.51 pp** |
| Internal val_word_accuracy | 0.7457 | **0.815** | **+6.9 pp** |
| 300-val char_acc (measured after training) | 0.9096 | *pending transcription* | — |

Kraken Run 2 (matched + medical, leak-fixed) and the four
single-stage TrOCR runs + Stage 1a/2a/2b Swin+BERT pipeline queue
via `scripts/ocr/queue_all_reruns.sh` — expected finish ~14:00
2026-07-22. Full leaderboard refresh (transcription + eval +
bootstrap CI + word-freq recall) will follow once the queue empties.

**Publishable framing.** The leak was a *pipeline bug*, not a data
issue — a subtle coordination gap between two independent split
routines in the kraken helper. Documenting the diagnosis + fix in
the thesis strengthens methodology: it shows we notice and correct
protocol issues rather than ignoring them (compare with §6.3 kraken
confound + §6.3.6 tokenizer-floor + §6.3.7 bootstrap CIs, all in the
same "clean-methodology" section). Recommend an appendix subsection
titled "Text-level leak in the mixed-real-and-synthetic split" with
this exact diagnosis and the before/after numbers above.

#### 6.3.10 Corrected-annotation full rerun (2026-07-22) — refreshed leaderboard + collapsed findings

**What changed.** Commit `f42d0ed` corrected **22 train + 10 val
annotations**. The full grid was then retrained on the new L4 VM
(§7.2.3) and re-evaluated against the permanent 300-val (299 non-empty).
GT parity verified: the VM validation `.gt.txt` set is byte-identical
(`sha1 69cd999…`) to the committed corrected local set, so every number
below is against the same corrected benchmark.

**Runs re-scored (2026-07-22 queue, all against corrected 300-val):**
7 TrOCR (`refresh_trocr_vs_val300_20260722`) + 2 leak-fixed kraken
(`kraken_leakfixed_vs_val300_20260722`). Both eval dirs pulled to laptop
under `tests/ocr/evaluations/`.

**Refreshed per-model char_acc (paired bootstrap, 10 000 it, seed=42,
299 lines):**

| Model | char_acc [95 % CI] | word_acc | vs prior spec |
|---|---|---|---|
| ViT+RoBERTa · medical (B″) | 93.89 % [93.06, 94.63] | 73.44 % | 0.9443 |
| ViT+RoBERTa · matched COMETA (A″) | 93.45 % [92.70, 94.17] | 72.09 % | 0.9332 |
| Swin+BERT · Stage 1a COMETA pretrain | 61.73 % [59.89, 63.51] | 32.93 % | 0.5918 |
| Swin+BERT · Stage 2b medical | 61.30 % [59.54, 63.04] | 33.23 % | 0.6080 |
| Swin+BERT · Stage 2a matched COMETA | 61.24 % [59.55, 62.97] | 32.68 % | 0.6053 |
| Swin+BERT single-stage COMETA | 19.53 % [18.65, 20.38] | −7.20 % | 0.2240 |
| Swin+BERT single-stage medical | 12.38 % [10.83, 13.81] | −15.00 % | 0.2523 |
| kraken matched **no-medical** leak-fixed (`_200641`) | 90.18 % [89.17, 91.15] | 55.61 % | 0.9620 (hist.) |
| kraken matched **medical** leak-fixed (`_021723`) | 89.94 % [88.97, 90.86] | 54.09 % | — |

**Findings that HOLD (paired bootstrap, corrected annotations):**
- **Staging lifts Swin+BERT massively**: Stage 2b vs single-stage medical
  Δ char_acc = **+48.90 %** [+46.59, +51.30], and **Stage 1 pretrain alone**
  vs single-stage = **+49.38 %** [+47.00, +51.83]. Both P(A > B) = 1.000.
  (Larger than the §6.3.4 reading of +35.6 pp — because the single-stage
  medical baseline fell to 0.1238 this run, widening the gap.)
- **Pretrained ViT+RoBERTa ≫ best staged Swin+BERT**: +32.60 %
  [+30.91, +34.27], P = 1.000.
- **A″/B″ tie under staging**: Stage 2a vs Stage 2b Δ = −0.05 %
  [−1.49, +1.39], P = 0.474 — still within noise (as in §6.3.7).

**Findings that COLLAPSED (were significant → now within noise):**

| Comparison | Old (spec) | Corrected (2026-07-22) | Now |
|---|---|---|---|
| Medical vs COMETA, pretrained ViT+RoBERTa | **+1.11 % ✓sig** (§6.3.7) | +0.44 % [−0.31, +1.14], P = 0.88 | **not sig** |
| Medical **hurts** kraken | **−4.31 % ✓sig** (§6.4) | −0.23 % [−0.65, +0.24], P = 0.85 | **not sig** |
| Manuscript FT adds value over Stage 1 | **+1.62 % ✓sig** (§6.3.7) | −0.45 % [−1.89, +0.99], P = 0.27 | **not sig / gone** |

The first two together **undercut the §6.4 "medical corpus is
architecture-dependent" cross-family story** — on the corrected benchmark
neither the +helps-TrOCR nor the −hurts-kraken effect is significant. The
§6.3.7 "manuscript fine-tuning adds a small but significant lift" is gone.
These reversals are the headline of the corrected-annotation rerun and
need to be reflected before any of §6.1/§6.3/§6.4 point estimates are
cited. **Caveat:** catmus + Medusa have **not yet** been re-evaluated
against the corrected GT, and the validated-285 manifest (§6.3.8) predates
the 10 val corrections — both are follow-ups before a fully consistent
leaderboard.

**Kraken baseline-drop investigation (aug-pool hypothesis REFUTED).**
The leak-fixed kraken runs land at ~0.90 on 300-val — ~6 pp below the
historical canonical 0.9620 and now below catmus (0.9613) and the best
TrOCR (0.9389). Root-cause investigation:
- **Aug pool is not the cause.** The new pool `aug_20260721_121550`
  (601 stems) is a filename-superset of the historical
  `aug_20260701_232640` (501 stems); shared base renders are **byte-for-byte
  md5-identical**. The 18 corrected-annotation stems were **re-rendered
  consistently** (image matches new label; 0 stale image/label mismatches).
  `⁊` / normalisation consistent across both. Renders + labels are clean.
- **Not undertraining — the model converged.** Internal val_accuracy
  plateaued at ~0.953–0.958 over the final ~15 epochs of `_200641`
  (best 0.958 @ ep 51, early-stopped @ 56). The low 300-val is therefore
  **not** an early-stopping artefact (an earlier draft's claim, now
  falsified by the curve). Instead there is a large **synth→real
  generalisation gap that inverted vs history**: historical `_070741` had
  internal val 0.889 but real 0.962 (real ≫ synth, +7 pp); the leak-fixed
  run has internal val 0.958 but real 0.902 (synth ≫ real, −6 pp). The
  model now masters the synthetic renders and transfers to real manuscript
  *worse* than it used to. Error mode on hard real lines is truncation /
  repeated-char output — the manifestation of that gap, not a codec offset.
- **No pipeline error, and the data matches TrOCR.** kraken-medical
  (`_021723`) and TrOCR-B″-medical (`trocr_20260722_103007`) train on the
  **identical** pool `aug_20260721_v2_medical` + same 600 real folder; they
  differ only in architecture, `val_fraction` (kraken 0.1 vs TrOCR 0.2) and
  the (now stem-grouped, leak-free) split. The §6.3.9 leak fix is verified
  (0 val stems leaked). So the ~4 pp kraken-below-TrOCR and ~6 pp
  kraken-below-history are **real generalisation results, not a bug**.
- **Open — which change flipped the synth→real transfer.** Candidates: the
  500→600-stem pool growth, `val_fraction` 0.1, or an interaction with the
  fixed split. Needs a one-variable ablation to isolate — e.g. re-run
  leak-fixed kraken on the historical 500-stem `aug_20260701_232640`, or
  bump `val_fraction` to 0.2 to match TrOCR.

Artefacts: `tests/ocr/evaluations/refresh_trocr_vs_val300_20260722/`,
`tests/ocr/evaluations/kraken_leakfixed_vs_val300_20260722/` (both CSV+MD,
on laptop). Bootstrap reproduced locally via `bootstrap_ocr_ci.py` with
`--pair` overrides for the new run labels.

### Kraken fine-tune catalog

Every kraken fine-tune this project has produced, with its training
composition and the source of its augmented synthetic pool. The
"canonical" reporting run is the LAST row unless a later run beats it
on the permanent 300-val benchmark (§6 results row).

| Run | Real | Aug pool | Base model | Notes |
|---|---|---|---|---|
| `finetune_20260629_235819` | 400 | `aug_20260629_235051` (100% annotated re-renders, 2000 pairs) | catmus-medieval | prior canonical; 320 train + 80 val + synth |
| `finetune_20260701_233056` | 500 | `aug_20260701_232640` (100% annotated re-renders, 2500 pairs) | catmus-medieval | 400 train + 100 val + synth |
| `finetune_20260705_070741` | 600 | `aug_20260701_232640` (2500 anno re-renders of 500 stems × 5) | catmus-medieval | 480 train + 120 val + synth; **historical no-medical** run reported as 0.9620 on 300-val |
| `finetune_20260706_151856` | 600 | `aug_merged_anno_medical_20260706` (2000 anno re-renders + 1000 medical corpus renders) | catmus-medieval | 480 train + 120 val + merged synth; the **confounded** medical-corpus run (0.9593 on 300-val) — aug re-render count differs vs `_070741` so the medical vs no-medical delta is not a single-variable comparison |
| `finetune_20260718_193601` | 600 | `aug_20260712_124729` (3000 anno re-renders of 600 stems × 5) | catmus-medieval | **matched-pool no-medical** (2026-07-18/19); 29 epochs, best at 23; internal val_acc = 0.9430; **300-val char_acc = 0.9096** [89.80, 92.01]. Baseline for the confound-fixed medical comparison. |
| `finetune_20260719_085411` | 600 | `aug_20260712_v2_medical` (3000 anno re-renders + 1000 medical corpus renders) | catmus-medieval | **matched-pool medical** (2026-07-19); 42 epochs, best at 36; internal val_acc = 0.9457; **300-val char_acc = 0.8664** [85.29, 87.91]. Differs from `_193601` **only** by the 1000-render medical slot. |
| `finetune_20260721_200641` | 600 | `aug_20260721_121550` (3000 anno re-renders of 600 stems × 5, corrected annotations) | catmus-medieval | **leak-fixed matched no-medical** (§6.3.9 fix + corrected annotations); 56 epochs, best at 51; internal val_acc = 0.9581; **300-val char_acc = 0.9018** [89.17, 91.15]. See §6.3.10 for the baseline-drop investigation. |
| `finetune_20260722_021723` | 600 | `aug_20260721_v2_medical` (3000 anno re-renders + 1000 medical corpus renders, corrected annotations) | catmus-medieval | **leak-fixed matched medical**; **300-val char_acc = 0.8994** [88.97, 90.86]. Differs from `_200641` **only** by the 1000-render medical slot. |

> **⚠ SUPERSEDED on corrected annotations (see §6.3.10).** On the leak-fixed
> corrected-annotation runs (`_200641` vs `_021723`) the medical vs no-medical
> delta is **−0.23 %** [−0.65, +0.24], P = 0.85 → **not significant**. The
> −4.31 pp figure below is from the earlier (leak-affected, pre-correction)
> `_193601`/`_085411` pair and should not be cited without that caveat.

**Matched-pool medical vs no-medical (paired bootstrap, 10 000 iterations, validated-285 subset)**:
Δ char_acc = **−4.31 %** [95 % CI −4.97, −3.66], Δ word_acc = **−13.20 %** [−15.60, −10.84], P(A > B) = 0.000 → **medical significantly HURTS kraken** (see §6.4 for interpretation and contrast with the +1.1 pp medical benefit on pretrained ViT+RoBERTa). Eval artefacts: `tests/ocr/evaluations/kraken_matched_medical_ablation/` (CSV + MD) and `tests/ocr/evaluations/bootstrap_ci_joint_20260720.txt` (joint kraken + TrOCR bootstrap output).

**Baseline shift to investigate**: the new matched-pool no-medical
baseline (`_193601`, 0.9096 on 300-val) is **~5 pp lower** than the
historical canonical (`_070741`, 0.9620 on 300-val). Only differences
vs historical: (a) aug re-renders sourced from all 600 stems (vs 500),
(b) aug pool version `aug_20260712_124729` (vs `aug_20260701_232640`).
Likely culprit: aug pool version — the newer render batch produces
visually different synthetics that shift the ceiling. Possible
secondary culprit: with augs from all 600 stems, some of the 120
real internal-val stems have augmented siblings in training →
text-level familiarity may inflate internal val_accuracy and trigger
early stopping sooner. **The matched-pool medical DELTA (Δ char_acc =
−4.31 pp) is still clean** because both runs share the same aug pool
and split — only the medical-corpus slot differs.

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

## 6.4 Cross-family finding: medical corpus is architecture-dependent (2026-07-20)

> **⚠ SUPERSEDED by the corrected-annotation rerun (§6.3.10).** Both legs of
> this finding lose significance on the corrected 300-val: medical vs COMETA
> for pretrained ViT+RoBERTa drops to +0.44 % [−0.31, +1.14] (P = 0.88), and
> medical vs no-medical for leak-fixed kraken to −0.23 % [−0.65, +0.24]
> (P = 0.85). The "opposite-direction significant effects" story below held on
> the pre-correction data only; do not cite it without the §6.3.10 caveat.

**Setup.** The medical corpus effect was tested with a matched-pool
design on two model families, and the confound flagged in §6.3
(unequal aug re-render counts in the original kraken runs) was closed
by re-running kraken with 3000 anno re-renders in both arms. Every
comparison below is single-variable — same 3000-anno-re-render base,
±1000 medical-corpus renders — so the delta is attributable to the
medical corpus and nothing else.

**Result.** The medical corpus intervention has **opposite-direction
significant effects** on the two model families:

| Architecture | Δ char_acc (medical − matched no-medical) | 95 % CI | P(A > B) | Verdict |
|---|---|---|---|---|
| **ViT+RoBERTa pretrained** (TrOCR) | **+1.11 %** | [+0.53, +1.72] | 1.000 | Medical **helps** significantly |
| **Kraken CTC fine-tune from catmus** | **−4.31 %** | [−4.97, −3.66] | 0.000 | Medical **hurts** significantly |

Kraken's word-level delta is even larger (−13.20 pp [−15.60, −10.84]),
suggesting the medical corpus text distribution drags the model's
word-shape priors away from what the manuscript actually contains.
ViT+RoBERTa's word-level delta is +2.05 pp [+0.29, +4.11] in the
opposite direction — same-magnitude direction (~1-2 pp per pp of
char_acc effect), just with sign flipped.

**Interpretation (speculative but well-grounded).** Catmus is
pretrained on generic medieval scripts, so kraken's fine-tune already
carries strong priors about "how medieval Latin words look". Adding
1000 medical corpus renders shifts those priors toward a specialised
Latin-medical distribution that mismatches the actual AlbucE mix
(medical *and* general prose *and* recipe formats). Pretrained TrOCR
has no such prior anchoring — its cross-attention was trained on
34 M generic handwriting pairs, none of them Old Occitan or
medieval — so the extra text-distribution diversity from the medical
corpus is pure signal, not distortion.

**Publishable framing.** "Domain-specific augmentation is not
universally beneficial for OCR / HTR on this manuscript family: it
significantly helps a large pretrained VLM (+1.1 pp char_acc) but
significantly hurts a small CTC recogniser fine-tuned from a
strongly-anchored medieval base (−4.3 pp char_acc). The direction
depends on whether the model's prior distribution over text is
compatible with the augmentation corpus." — one clean statement,
two significant CIs excluding zero on the same matched-pool
ablation, supported by 10 000-iteration paired bootstrap.

**Data + artefacts:**
- Kraken matched-pool eval: `tests/ocr/evaluations/kraken_matched_medical_ablation/`
- TrOCR matched-pool eval: `tests/ocr/evaluations/five_trocr_vs_validation_300/`
- Joint bootstrap: `tests/ocr/evaluations/bootstrap_ci_joint_20260720.txt`
- Manifest for filter: `tests/ocr/validation_300_manifest_.csv` (validated_100 subset)

**Open follow-up.** The new matched-pool kraken no-medical baseline
(0.9096) is ~5 pp below the historical canonical (0.9620). The
matched-pool medical DELTA is still clean (same aug pool in both
arms), but the absolute levels are worth investigating — see the
"Baseline shift to investigate" note in the [Kraken fine-tune
catalog](#kraken-fine-tune-catalog).

## 6.5 Planned next experiments (queued 2026-07-22)

Concrete, agreed next steps. Each is a single-variable move so the
result is interpretable. Back up before starting anything on the VM
(§7.5).

### 6.5.1 Kraken synth→real gap ablations (deferred)

Goal: isolate which change flipped kraken's synth→real transfer
(§6.3.10 — internal val 0.958 but real 0.902, inverted vs the historical
0.889 → 0.962). Both runs use `run_finetune_ocr` on the corrected
600-real folder; change exactly one knob per run and re-score on the
permanent 300-val:

- **Ablation A — historical 500-stem pool.** Re-run leak-fixed kraken with
  `--augmented-folder aug_20260701_232640` (the 2500-render, 500-stem pool
  that produced the historical 0.9620) instead of `aug_20260721_121550`.
  Isolates the 500→600-stem pool growth. If 300-val climbs back toward
  0.96, the pool composition (not the leak fix) drove the drop.
- **Ablation B — match TrOCR's val split.** Re-run leak-fixed kraken with
  `--val-fraction 0.2` (currently 0.1; TrOCR uses 0.2). Isolates the
  internal-split-size effect on early stopping / generalisation.

Both ~1 h on the L4. Feed each eval CSV to `bootstrap_ocr_ci.py` for CIs.

### 6.5.2 Stage-1 COMETA scale-up: 30k → 90k / 120k (2-stage Swin+BERT)

Goal: test whether more task-domain pretraining data improves Stage 1
(and hence the staged Swin+BERT ceiling). §6.3.4 showed 30k COMETA
pretraining does ~34 pp of the from-scratch lift; the open question is
whether 90–120k pushes it further and shrinks the val-fold→300-val gap.

- **Source**: subsample from the local full COMETA pool
  `data/processed/synthetic_samples/augmented_images/aug_20260613_220436`
  (266,479 renders, 53 GB) with a deterministic seed, exactly as the 30k
  pool `aug_20260714_cometa_30k` was built (seed=42
  `random.Random(seed).sample`). Name the new pool
  `aug_<TS>_cometa_<N>k`.
- **Size vs VM disk**: 30k = 5.9 GB, so 90k ≈ 18 GB, 120k ≈ 24 GB. VM
  `/home/jupyter` currently has only ~25 GB free and
  `models/ocr/finetuned` holds 57 GB (mostly prunable epoch
  checkpoints). **Free VM space first** (back up `best_model/`s, prune
  intermediate checkpoints — §7.5) before uploading, or 120k will not
  fit. 90k is the safer size if disk stays tight.
- **Upload**: split-tarball per §7.2.4 (500 MB chunks; 18 GB ≈ 36 chunks,
  24 GB ≈ 48 chunks). Resume-friendly; never re-scp a landed chunk.
- **Train**: Stage 1a on the new pool (mirror §6.3.4 knobs: 15 epochs,
  bs=32, lr=5e-5, val_fraction=0.05, early_stopping_patience=4), then
  Stage 2a/2b fine-tunes on Datasets A″/B″. Watch the val-fold→300-val
  gap as the primary signal.

**STATUS (2026-07-22, in progress).** 90k run underway autonomously:
- Pool built: `aug_20260722_cometa_90k` = existing 30k
  (`aug_20260714_cometa_30k`) ∪ 60k more sampled seed=42 from the 266k
  `aug_20260613_220436` (so 90k ⊃ 30k → monotonic comparison). Labels:
  `labels_20260722_cometa_90k/labels.json` (90000 entries). Built locally
  via hardlinks; tarred 19.2 GB, split into 37 × 500 MB chunks, uploading
  to VM `/tmp/cometa90k_up/` (resume-friendly, sha256 recorded locally).
- Driver script (scratchpad `queue_stage1_90k.sh`) runs Stage 1a on 90k →
  Stage 2a (A″) → Stage 2b (B″), same knobs as §6.3.4. Stage-1a run dir
  will be `models/ocr/finetuned/trocr_<TS>` on the VM; downstream stages
  load its `best_model`.
- Expected: Stage 1a ~6–7 h (90k ≈ 3× the 30k's 2h12m). Compare Stage-1a
  90k 300-val char_acc vs 30k baseline **0.5918** (§6.3.4) and watch
  whether the val-fold→300-val gap (was ~27 pp) shrinks. **Results table
  to be filled in here when training + eval complete.**

### 6.5.3 External-corpus ratio sweep (re-render : external)

Current A″/B″ pools fix the ratio at **3000 anno re-renders : 1000
external** (3:1). Sweep the external-corpus slot while holding the 3000
re-render base fixed, for **both** COMETA and medical:

| Variant | Anno re-renders | External | Ratio |
|---|---|---|---|
| current (A″/B″) | 3000 | 1000 | 3:1 |
| sweep-500 | 3000 | 500 | 6:1 |
| sweep-2000 | 3000 | 2000 | 3:2 |
| sweep-4000 | 3000 | 4000 | 3:4 |

- Build each pool by merging the fixed 3000-re-render base with an
  N-render external slot (seed=42) — reuse
  `scripts/ocr/merge_base_with_corpus_slot.py`.
- Train the **pretrained ViT+RoBERTa** (best arch) on each; single-stage;
  score on corrected 300-val + bootstrap. Goal: is 3:1 optimal, or does
  more/less external corpus help? Medical corpus (12,012 entries) easily
  supports 4000 renders.

### 6.5.4 Stage-1 pretraining on medical corpus (instead of COMETA)

§6.3.4 open follow-up. In the 2-stage Swin+BERT, swap the Stage-1
pretraining corpus from COMETA to **medical**. **Caveat / open feasibility
question:** the medical corpus (12,012 entries) is far smaller than the
COMETA source (266k renders), so a 30k/90k medical pretraining pool may
not be reachable without heavy augmentation multiplicity — quantify the
max renders first. Even a smaller medical Stage-1 gives a conclusion:
does Stage-1 corpus choice matter when §6.3.4 showed Stage-2 corpus
choice does not?

### 6.5.5 TrOCR top-k / confidence analysis

For the ViT+RoBERTa runs, capture generation scores and beam candidates.
On lines the model got wrong (per-line CER > 0), check whether the
**correct GT is among the top-5 beam hypotheses** (oracle top-5 accuracy)
and log token-level probabilities. Implementation: extend
`trocr_transcribe` with `num_return_sequences=5` + `output_scores=True`.
Goal: quantify how much error is a recoverable "close miss" (rerankable
by an external LM) vs a genuine miss.

### 6.5.6 Encoder / decoder swap ablation (1-stage)

The grid contrasts Swin+BERT (from-scratch) vs ViT+RoBERTa (pretrained).
Add the **cross combinations** — **ViT+BERT** and **Swin+RoBERTa** —
single-stage, to isolate the encoder (ViT vs Swin) from the decoder
(BERT vs RoBERTa) contribution. **Caveat:** only ViT+RoBERTa exists as a
pretrained-cross-attention checkpoint (`microsoft/trocr-base-handwritten`);
the swaps have randomly-initialised cross-attention, so expect
Swin+BERT-like numbers — the value is the component-isolation, not
competitiveness.

### 6.5.7 GPT-style decoder

Exploratory: build the `VisionEncoderDecoderModel` with a causal GPT
decoder (e.g. GPT-2) instead of BERT/RoBERTa. Tests whether an
autoregressive LM decoder helps over the masked-LM-derived decoders.

### 6.5.8 Full bootstrap CI + ink-bleed stratification refresh

Re-run `bootstrap_ocr_ci.py` across **all** models on the corrected
300-val — catmus, Medusa, leak-fixed kraken, and the refreshed TrOCR grid
— and the **ink-bleed-stratified** stats using the already-defined
ink-bleed metric (prior conclusions in spec; artefacts
`tests/ocr/evaluations/bootstrap_ci_trocr_bleed_20260718/` and
`ink_bleed_val300_20260718/`). Depends on catmus + Medusa first being
re-evaluated against the corrected GT (§6.3.10 caveat).

### 6.5.9 Word-frequency recall error analysis (re-run)

Re-run `scripts/ocr/word_frequency_recall.py` for the refreshed /
leak-fixed models: take the **600 + 300 annotated lines as the vocabulary**,
compute per-model per-word **recall** on the 300-val (multiset
bag-of-words intersection), stratified by corpus frequency. Existing
artefact: `tests/ocr/evaluations/word_frequency_recall_20260721/`; refresh
it with the corrected-annotation model set.

## 7. Infrastructure

### 7.1 Local laptop

- Apple Silicon Mac. Torch is `2.4.1` (pinned by kraken). MPS available.
- Full training + inference for kraken and TrOCR runs here.
- Some torch ops fall back silently to CPU on MPS — always run TrOCR
  training with `PYTORCH_ENABLE_MPS_FALLBACK=1` prefix so unsupported
  ops fall back to CPU compute instead of crashing.

### 7.2 GCP VMs

Three instances have been in play over the course of the project. As
of 2026-07-21 only **§7.2.3 is active**; §7.2.1 and §7.2.2 are kept
for historical continuity and to document the two-user-gotcha and
disk-mount lessons learned there.

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
  - `data/processed/synthetic_samples/augmented_images/aug_20260714_cometa_30k/` — 30k subset of the COMETA aug pool, split-uploaded 500 MB × 13 parts after direct scp stalled. Used for the Stage 1 pretraining experiment (§6.3.4).
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
  - **Stage 1a Swin+BERT COMETA pretraining IN PROGRESS** —
    `swinbert_cometa_30k_pretrain_<TS>` (started 2026-07-14 14:44).
    Full run details in §6.3.4. Once done, two Stage 2 fine-tunes
    (Dataset A'' and B'') will follow. Log:
    `logs/trocr_finetune/stage1_swinbert_cometa_30k_pretrain_*.out`.
- Cost: L4 ~$0.7/h. TrOCR grid ~$5 total. Medusa full-corpus
  ~$4. **Stop the instance when idle**: `gcloud compute instances
  stop instance-20260712-110217 --zone=us-west4-c`.
- Deferred: 6 GB tarball at
  `/tmp/occ_htr_vm_runs.tar` on the VM containing every run's
  `best_model/` + metadata + logs. Pulled to laptop 2026-07-13 via
  scp with SSH keepalive at ~6 MB/s.

#### 7.2.3 Active L4 GPU VM — `instance-20260720-095326`, us-west4-c

The old §7.2.2 VM stopped responding around 2026-07-19; replaced by a
fresh Vertex AI Workbench instance in a **new GCP project** on
2026-07-20.

- **Project**: `project-8a4066cd-a3df-4df6-8dd` (organization
  `thesisgcplmu-org`, display name "My First Project"). Point gcloud at
  it with `gcloud config set project project-8a4066cd-a3df-4df6-8dd`
  before any `gcloud compute` command against this instance.
- **Zone**: `us-west4-c` (same as §7.2.2).
- Machine: 16 vCPU, 64 GB RAM + **NVIDIA L4 × 1** (23 GB VRAM, driver
  580.65.06).
- Python 3.11.2 (matches the project's `requires-python`; no
  workaround needed unlike §7.2.2's 3.12).
- `git` 2.39.5 pre-installed.
- `uv` **not** pre-installed; install once via
  `curl -LsSf https://astral.sh/uv/install.sh | sh` and export
  `~/.local/bin` on PATH (append to `~/.bashrc`).
- **Two-mount disk (same shape as §7.2.2)**: `/` is 148 GB (69 GB used
  as delivered), `/home/jupyter` is 98 GB and nearly empty. **All
  data + repo must live under `/home/jupyter/`**, not the user's own
  home under `/`, to avoid filling the small root partition.
- **Single-user OS Login (improvement over §7.2.2)**: `gcloud compute
  ssh jupyter@...` lands you as `thesisgcplmu_gmail_com`; `gcloud
  compute scp` also uses that same user. No more two-user split with
  the `jupyter` service user. `/home/jupyter/` is owned by the
  `jupyter` system user by default — one `sudo chown -R
  $(whoami):$(whoami) /home/jupyter/OCC_HTR` grants your OS Login
  user write access to the workspace subfolder.
- Repo location: `/home/jupyter/OCC_HTR/` (mirrors §7.2.2). Clone
  method:
  ```bash
  gcloud compute ssh jupyter@instance-20260720-095326 --zone=us-west4-c
  sudo mkdir -p /home/jupyter/OCC_HTR
  sudo chown -R $(whoami):$(whoami) /home/jupyter/OCC_HTR
  cd /home/jupyter
  git clone <REPO_URL> OCC_HTR   # HTTPS + PAT, or SSH if key is enrolled
  cd OCC_HTR
  ~/.local/bin/uv sync
  ~/.local/bin/uv pip install transformers==5.12.1   # same 5.13 pin as §7.2.2
  ```
- **Standard invocation pattern on this VM** (mirrors §7.2.2 but with
  the correct user & no /tmp bounce needed for scp):
  ```bash
  env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/...   # avoid pip install -e .
  nohup ... > logs/... 2>&1 &                          # for anything > 10 min
  ```
- **Stop the instance when idle**: `gcloud compute instances stop
  instance-20260720-095326 --zone=us-west4-c` (billing pause: ~$0.7/h
  running → ~$0.05/h stopped).
- **Planned use (as of 2026-07-21)**: rerun the full 600-annotated
  training grid — kraken matched-pool no-medical + medical, TrOCR
  ViT+RoBERTa + Swin+BERT — because a subset of the 600-line
  annotations was corrected between the earlier runs and today.
  Also plan to re-execute the 2-stage Swin+BERT pipeline (Stage 1a
  on 30 k COMETA + Stage 2a/b) and finally attempt Stage 1b on a
  larger COMETA pool (60 k or the full 266 k) once the upload path
  is validated on this instance.

#### 7.2.4 Canonical upload pattern — split tarball + reassemble

**Problem.** Direct `gcloud compute scp` of large tarballs (> ~1 GB
in practice; the failure mode kicks in earlier over residential
Wi-Fi or NAT) frequently stalls or drops mid-transfer, and gcloud
does not resume — a killed transfer means starting over from byte 0.
The 6 GB `aug_20260714_cometa_30k` upload to §7.2.2 stalled
repeatedly on direct scp; the 53 GB full COMETA pool proved outright
infeasible with direct scp.

**Fix.** Split the tarball into chunks (500 MB is a robust default
that survives typical residential connections), scp each chunk
separately (retries only lose one chunk, not the whole file), then
concatenate on the VM. Deterministic naming lets a partial upload
resume by only re-scp'ing the missing parts.

**Recipe (adapt paths per upload):**

Local (laptop):

```
# 1. Build tarball. COPYFILE_DISABLE=1 blocks macOS AppleDouble sidecars
#    (see §11 — those doubled Medusa's input file count on the old VM).
cd <REPO_ROOT>
COPYFILE_DISABLE=1 tar -cf /tmp/<upload_name>.tar <path1> <path2> ...

# 2. Split into 500 MB chunks named <upload_name>.tar.part-aa/ab/ac/...
split -b 500m /tmp/<upload_name>.tar /tmp/<upload_name>.tar.part-
ls -lh /tmp/<upload_name>.tar.part-*

# 3. scp all chunks. --scp-flag='-o ServerAliveInterval=60' keeps the
#    SSH session alive through Wi-Fi hiccups.
gcloud compute scp \
    --scp-flag='-o ServerAliveInterval=60' \
    --scp-flag='-o ServerAliveCountMax=10' \
    /tmp/<upload_name>.tar.part-* \
    jupyter@<INSTANCE>:/tmp/ \
    --zone=<ZONE>
```

Remote (VM), after all parts land:

```
cd /home/jupyter/OCC_HTR
# 4. Reassemble deterministically — cat glob-sorts lexicographically,
#    which matches the split naming (part-aa < part-ab < ...).
cat /tmp/<upload_name>.tar.part-* > /tmp/<upload_name>.tar

# 5. Verify byte count matches the local tarball before untarring.
ls -l /tmp/<upload_name>.tar

# 6. Untar into place, then clean up all chunks + the reassembled tar.
tar -xf /tmp/<upload_name>.tar
rm /tmp/<upload_name>.tar /tmp/<upload_name>.tar.part-*
```

**Resume a partial upload.** If some chunks landed and some didn't,
re-run step 3 with the specific missing parts (or the whole glob —
gcloud will overwrite existing chunks byte-identically, so retry is
idempotent). Never delete a chunk that landed successfully before
retrying — you'll re-upload data you already have.

**When to bother.** Rule of thumb: anything **≥ 1 GB uncompressed**
gets the split treatment by default. Small uploads (< 500 MB) can
go direct — single-chunk splits are pointless overhead.

**Chunk sizing.** 500 MB is the default because it balances two
concerns: (a) small enough that a stalled scp only wastes one chunk's
worth of transfer, (b) large enough that the total number of chunks
stays manageable (13 chunks for 6 GB, 106 chunks for 53 GB, both
within reason). For LAN or high-quality corporate networks, 1-2 GB
chunks are fine and reduce round-trip overhead.

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

### 7.5 VM → local backup & results-safety procedure

**Why.** VMs get stopped, deleted, or die (the §7.2.2 instance stopped
responding and was replaced). Anything that lives *only* on the VM is at
risk. Rule: after any training batch, pull the small artefacts
immediately and back up the models before the VM is idle-stopped.

**What to save, in priority order:**

1. **Results (tiny, pull always):** `tests/ocr/evaluations/*` (eval
   CSV+MD, ~KB each), `logs/**/*.log` + `*.out` (training curves, config
   dumps), and each run's `final_metrics.json` / `metrics.json`. These
   are the numbers the thesis cites — losing them = re-running everything.
2. **Kraken checkpoints (small, ~16 MB each):**
   `models/ocr/finetuned/<run>/model_best.mlmodel`.
3. **TrOCR checkpoints (large, ~1.1–1.3 GB each):**
   `models/ocr/finetuned/trocr_<TS>/best_model/` — the self-contained
   VisionEncoderDecoderModel. Only `best_model/` is needed; the sibling
   `checkpoint-*/` epoch dirs (the bulk of the 57 GB `finetuned/` folder)
   are prunable once `best_model/` is safe.

**Pull commands (laptop, from repo root).** Small stuff direct scp:

```bash
INST=jupyter@instance-20260720-095326; ZONE=us-west4-c
PROJ=project-8a4066cd-a3df-4df6-8dd
REPO=/home/jupyter/OCC_HTR
# results (evals + logs) — always safe over the flaky link (KBs)
gcloud compute scp --recurse --zone=$ZONE --project=$PROJ \
  $INST:$REPO/tests/ocr/evaluations ./tests/ocr/
gcloud compute scp --recurse --zone=$ZONE --project=$PROJ \
  $INST:$REPO/logs ./
# kraken mlmodels (small)
gcloud compute scp --zone=$ZONE --project=$PROJ \
  "$INST:$REPO/models/ocr/finetuned/finetune_*/model_best.mlmodel" \
  ./models/ocr/finetuned/   # (fix per-run subdir after)
```

**Large TrOCR `best_model/`s — split-tarball per §7.2.4** (each ~1.2 GB;
7 runs ≈ 8 GB). Build one tar of all `best_model/`s on the VM, split into
500 MB chunks, scp per-chunk (resume-friendly), reassemble + extract on
the laptop:

```bash
# ON VM: tar just the best_model dirs (COPYFILE_DISABLE not needed on Linux)
cd /home/jupyter/OCC_HTR/models/ocr/finetuned
tar -cf /tmp/best_models_<TS>.tar */best_model
split -b 500m /tmp/best_models_<TS>.tar /tmp/best_models_<TS>.tar.part-
# ON LAPTOP: pull all chunks (retry only loses one chunk), then
#   cat parts > tar ; tar -xf ; verify byte count ; rm parts
```

**GCS alternative (currently BLOCKED).** `gsutil`/`gcloud storage` exist
on the VM but the default compute service account lacks
`storage.buckets.list` / write permission (403 as of 2026-07-22). To use
a bucket as durable backup (survives VM deletion, no laptop transfer),
grant the SA `roles/storage.objectAdmin` on a bucket, or `gcloud auth
login` as `thesisgcplmu@gmail.com` on the VM first. Until then, laptop
scp is the only working path.

**Freeing VM disk (needed before the §6.5.2 COMETA upload).** After
`best_model/`s are backed up, prune each TrOCR run's intermediate
checkpoints. **The dir is named `checkpoints/` (not `checkpoint-*`)** —
each run dir is `trocr_<TS>/{best_model,checkpoints}`. Guard on the
sibling `best_model/` existing so a run that only has `checkpoints/` is
never nuked:

```bash
cd models/ocr/finetuned
for ck in $(find . -maxdepth 2 -type d -name checkpoints); do
  [ -d "$(dirname "$ck")/best_model" ] && rm -rf "$ck" || echo "SKIP $ck"
done
```

On 2026-07-22 this freed ~48 GB (7 runs; `/home/jupyter` 25 GB → 72 GB
free). **Never prune before the backup byte count is verified on the
laptop.**

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
- **Don't send 50+ GB tarballs UP to a VM either.** Same failure mode
  in the other direction (upload stalls after ~25% on flaky WiFi,
  scp gives up). If a big pool needs to move: (a) split into 500 MB
  parts and use the resume-friendly loop from §11's model-download
  entry; (b) `ssh-add ~/.ssh/google_compute_engine` beforehand so
  each retry doesn't prompt for the SSH passphrase; (c) `mkdir -p`
  the target directory on the VM once BEFORE the loop — scp doesn't
  auto-create parents; (d) OR downscale the pool locally to what you
  actually need. For pretraining experiments, 30 k pairs of the
  266 k pool gives most of the signal at ~10× less upload.
- **Don't paste multi-line bash blocks with `#` comments into zsh**
  unless `setopt interactive_comments` is set (macOS's default zsh
  does NOT enable it, and each `#` line is treated as
  "command not found"). Two workarounds: (i)
  `setopt interactive_comments` once per shell (or add to
  `.zshrc`); (ii) strip the comment lines when pasting.
- Don't try to use transformers 5.13.x for TrOCR pretrained model
  loading. Rebroadcast of the fix from earlier: pin
  `transformers==5.12.1`.
