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

### 5.3 Synthetic augmented pool — `data/processed/synthetic_samples/augmented_images/aug_20260613_220436/`

- **266,478 PNGs** across **88,827** synthetic source stems (~3 aug
  variants per source line). Names like
  `Additional_10323_l00001_aug00.png`. Labels JSON:
  `data/processed/synthetic_samples/img_labels/labels_20260613_220436/labels.json`
  (266k entries).
- Source: rendered text from COMETA + other medieval corpora, image-
  augmented ×N via `augmentation_techniques.py`. Labels normalised via
  `correct_labels.py`.
- Kraken uses the full pool. TrOCR subsamples to
  `TROCR_MAX_AUG_SAMPLES=5000` by default because MPS + swin-base +
  mBERT can't chew through 266k images in tractable time.
- **Not to be confused with augmentations of the real photos** — these
  are augmentations of *synthetic renders*.

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

The eval every model is compared against for thesis reporting. Run:
`tests/ocr/evaluations/six_way_vs_validation_300/`. 299 lines scored
(1 gt intentionally empty).

| Model | CER | char_acc | WER | word_acc | median CER | median WER |
|---|---|---|---|---|---|---|
| catmus baseline | 0.0387 | 0.9613 | **0.1434** | **0.8566** | 0.0278 | 0.1250 |
| Medusa 0.2 Line 9B (cleaned v2) | 0.0490 | 0.9510 | 0.3106 | 0.6894 | 0.0435 | 0.2857 |
| kraken 400 real (`finetune_20260629_235819`) | 0.0420 | 0.9580 | 0.2358 | 0.7642 | 0.0286 | 0.2000 |
| kraken 500 real (`finetune_20260701_233056`) | 0.0390 | 0.9610 | 0.2188 | 0.7812 | 0.0278 | 0.1667 |
| **kraken 600 real** (`finetune_20260705_070741`) | **0.0380** | **0.9620** | 0.2144 | 0.7856 | 0.0278 | 0.1667 |
| kraken 600 real + medical (`finetune_20260706_151856`) | 0.0407 | 0.9593 | 0.2275 | 0.7725 | 0.0278 | 0.1667 |

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

### 6.3 TrOCR track (separate architecture family)

| Model | Train data | Val set | char_acc | word_acc | Notes |
|---|---|---|---|---|---|
| TrOCR Swin+BERT real-only, run `trocr_20260710_125139` | 480 real (val-fold split) | 120 real val | 0.2411 | 0.0000 | 480 lines can't teach 57M randomly-initialised cross-attn params; expected baseline |
| TrOCR Swin+BERT + aug (`trocr_20260710_142341`) | 600 real + 5000 aug subsampled (source-stem split, 5509 unique stems) | val-fold | *in progress* | — | started 2026-07-10 14:23; same aug pool as kraken; final numbers will land in the run's `final_metrics.json` |

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

### 7.2 GCP VM (Medusa inference only)

- Zone: `us-central1-a`.
- Instance name: `instance-20260629-174751`.
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

# 4) TrOCR fine-tune (Swin+BERT, MPS, defaults to 600 real + 5000 aug).
#    Set TROCR_AUGMENTED_FOLDER= TROCR_LABELS_JSON= for real-only.
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
- What CER does a `microsoft/trocr-base-handwritten` fine-tune reach
  vs. the from-scratch Swin+BERT? The pretrained cross-attention
  should close most of the gap; this validates or rules out the
  architecture direction.
- If the trocr-base experiment closes the gap to catmus, is there
  value in ensembling the top models for the thesis's final headline
  number? Defer until we have the numbers.

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
