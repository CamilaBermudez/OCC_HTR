# OCC-HTR

OCR/HTR for a medieval Occitan medical manuscript (AlbucE): from page photos
to line-level diplomatic transcriptions, compared against the scholarly
edition, with a human-in-the-loop review frontend.

- **[docs/project_organization.md](docs/project_organization.md)** — what the
  project is, repo layout, conventions (src/scripts/makefile pattern, run
  naming, data layout, evaluation protocol).
- **[docs/user_guide.md](docs/user_guide.md)** — how to do things: prepare
  ground truth, fine-tune (kraken / TrOCR), transcribe, align the scholarly
  edition, run the comparison pipeline, and the catalog of analysis scripts.
- **[docs/model_results.md](docs/model_results.md)** — consolidated model
  results on the 300-line validation set.
- **`spec.md`** — the running lab notebook: every experiment, decision and
  negative result, dated. Read it before repeating an experiment.

Quick start: `uv sync`, then `make frontend` to explore the manuscript viewer
at http://localhost:8000, or see the user guide for the pipeline targets.
