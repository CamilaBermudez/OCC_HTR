# OCC-HTR — project organization

What this repository is, how it is laid out, and the conventions everything
follows. Read this first; the hands-on companion is
[`docs/user_guide.md`](user_guide.md).

## What the project is

OCR/HTR for a medieval Occitan medical manuscript (the *AlbucE* corpus:
71 pages, ~13.7k text lines). The pipeline goes from raw page photographs to
line-level diplomatic transcriptions, compares them against the scholarly
edition, and closes the loop with a human-in-the-loop review frontend.
Recognisers covered: kraken/CATMuS (CTC) fine-tunes + a char-LM rescorer,
TrOCR-style ViT+RoBERTa fine-tunes, and frozen baselines (CATMuS-medieval,
Medusa 9B VLM, PaddleOCR).

## The three-layer code convention

| layer | contains | rule |
|---|---|---|
| `src/` | importable library code (`occ-htr` package) | pure functions/classes, no CLI parsing |
| `scripts/` | thin CLI wrappers around `src/` functions | argparse + logging only; one script = one task |
| `makefile` | the entry points | every routine task is a target; parameters are make variables with sane defaults, overridable on the command line |

So the way to run anything is `make <target> [VAR=value ...]`; the way to reuse
logic is `from src.<pkg> import ...`. Scripts that are pure *analysis* (run
once for a spec section) live in `scripts/ocr/` too but may not have a make
target — they are catalogued in the user guide.

Sub-packages of `src/`: `data_preprocessing` (segmentation → crops →
binarize → filter → ink-bleed), `data_augmentation` (synthetic medieval text +
rendering + augmentations), `ocr` (recognisers, LM rescoring, alignment,
diffing, evaluation), `tokenizer`, `utils`. `scripts/` mirrors this layout;
`scripts/cluster/` holds the SLURM job files for the GPU cluster.

## Run-directory and naming conventions

- **Every run is timestamped**: outputs go to `<name>_<YYYYMMDD_HHMMSS>/`
  (e.g. `models/ocr/finetuned/finetune_20260806_123435/`). Never overwrite or
  delete a finished run dir to reuse its name — make a new one.
- **Line stems** are globally unique: `<page>_line_<N>`
  (e.g. `06_f_001v_002_line_107`), where `<page>` is the manuscript page key.
  Everything joins on these stems.
- **Logs**: every long-running task writes to `logs/<area>/<run>.log`.
  Aggregate numbers (final tables, metrics) are *mirrored into the log*, so
  results are recoverable from `logs/` alone even if an artifact folder moves.
- **Analysis artifacts** (plots, CSVs, verdict files) go to
  `tests/ocr/evaluations/<analysis_name>/`, usually date-suffixed.
- **Provenance**: transcription/prediction dirs carry a `_provenance.json`
  (model, run, parameters) so predictions can always be traced to the model
  that produced them.
- **Models**: `models/ocr/` (kraken `.mlmodel`, incl. `finetuned/<run>/model_best.mlmodel`),
  `models/vit_*/<run>/best_model/` (HF checkpoints; `resize_mode.txt` inside
  records the image-prep mode the model was trained with), `models/layout/`
  (YOLO segmentation).

## Data layout (`data/`)

- `data/raw/` — immutable inputs: page photographs
  (`original_manuscript/reproduction14453_100`), the scholarly edition
  (`AlbucE.txt`), external corpora (COMETA, medical texts, Pansier).
- `data/processed/` — everything derived, one folder per pipeline stage:
  `img_layout` (masks) → `segmented_images` → `extracted_lines` →
  `binarized_images` → `filtered_images` (the **kept** line crops the whole
  project runs on: `filtered_images/<stamp>/original/kept/<page>/*.png`) →
  `transcription/<run>/` (per-page dirs of `<page>_line_<N>.txt` +
  `<page>_full.txt`) → `line_compare/` (viewer comparison JSON).
- `data/processed/annotated_samples/OCR/` — ground truth (see user guide for
  the format): `full_annotated/` = the 600-line training pool,
  `validation/` = the **permanent held-out 300-line benchmark. Never train or
  tune on it; it is report-only.** Batch folders (`batch_5`, `train500_*`, …)
  record how the pool was accumulated.

## The running-state document: `spec.md`

`spec.md` at the repo root is the canonical lab notebook: every experiment,
decision, negative result, and its artifact paths, in dated sections
(§-numbered). **Read it before repeating any experiment — settled negatives
(synthetic data for kraken, router ensembles, minim-only LM …) are recorded
there so they are not re-attempted.** New results are appended there, with the
convention that superseded numbers are corrected in place with a dated
"CORRECTION" note rather than silently edited. Consolidated final numbers live
in `docs/model_results.md` (+ `plot_model_results.py` to regenerate figures).

## Environment & tooling

- Python 3.11, managed by **uv**; `pyproject.toml` is the source of truth
  (`requirements.txt` is legacy). `uv sync` creates `.venv`;
  run things as `PROJECT_ROOT=. uv run python …` (the makefile does this).
- `make setup-precommit` installs the pre-commit hooks (ruff etc.).
- Heavy GPU work (TrOCR training, VLM inference) runs on a SLURM cluster via
  `scripts/cluster/*.sbatch`; models + logs are always pulled back locally
  after a run so evaluation never depends on the VPN.
- The viewer (`frontend/`, FastAPI + vanilla JS SPA) is launched with
  `make frontend` and reads the processed data paths above (overridable via
  `VIEWER_*` environment variables — see `frontend/config.py`).

## Evaluation protocol (project-wide rules)

- Report set = the 300-line validation set, untouched by any training/tuning.
  Hyper-parameters (LM λ, temperature, margins) are tuned on dev splits of the
  600 pool, then reported once on the 300.
- Metrics: corpus CER / char-acc + WER / word-acc, per-line medians, and 95%
  paired-bootstrap CIs (`bootstrap_ocr_ci.py`, 10 000×, seed 42). CER is
  clipped at 1 per line.
- Transcription style is **diplomatic** (base letters; abbreviations,
  u/v, i/j as written in the manuscript) — the scholarly edition is the
  *normalized* counterpart, which is why model-vs-edition differences are
  classified (editorial vs substantive) instead of counted raw.
