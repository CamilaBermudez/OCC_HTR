"""Path configuration for the manuscript-viewer frontend.

Every path is overridable via an environment variable so a new pipeline
run doesn't require editing code — set the ``VIEWER_*`` env, restart
the server. Defaults are the currently-canonical pipeline versions per
``spec.md`` §5-6.
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class Config:
    """Resolved-Path bundle passed to :class:`ManuscriptRepo`."""

    raw_pages_dir: Path
    """Folder of original ``.jpg`` manuscript pages (human-named)."""

    segmentation_dir: Path
    """Folder of ``<page_key>.json`` YALTAi/kraken segmentation outputs."""

    model_transcription_dir: Path
    """Root of ``<page_key>/<page_key>_line_<N>.txt`` from an OCR/HTR run.
    Change via ``VIEWER_MODEL_TRANSCRIPTION`` once a newer model is
    ready — the viewer will pick it up on restart."""

    scholarly_txt: Path
    """Per-line-aligned scholarly transcription file with headers like
    ``========== IMAGE: <page_key>_full ==========``."""

    filtered_kept_dir: Path
    """Filtered kept line crops (the folder the annotators saw)."""

    line_alignment_json: Path
    """Per-page content-based line alignment (model line -> scholarly line),
    produced by ``scripts/ocr/align_transcriptions.py``. Used to pair each
    segmentation line with the *correct* scholarly line instead of the
    positional guess (spec §6.6). Default sits next to the model transcription;
    override via ``VIEWER_LINE_ALIGNMENT``. Missing file => positional fallback."""

    @classmethod
    def from_env(cls) -> "Config":
        model_transcription_dir = Path(
            os.environ.get(
                "VIEWER_MODEL_TRANSCRIPTION",
                REPO_ROOT / "data/processed/transcription/finetune_400_full_corpus",
            )
        )
        return cls(
            raw_pages_dir=Path(
                os.environ.get(
                    "VIEWER_RAW_PAGES",
                    REPO_ROOT / "data/raw/original_manuscript/reproduction14453_100",
                )
            ),
            segmentation_dir=Path(
                os.environ.get(
                    "VIEWER_SEGMENTATION",
                    REPO_ROOT / "data/processed/segmented_images/segmentation_20260618_111517",
                )
            ),
            model_transcription_dir=model_transcription_dir,
            line_alignment_json=Path(
                os.environ.get(
                    "VIEWER_LINE_ALIGNMENT",
                    model_transcription_dir / "line_alignment.json",
                )
            ),
            scholarly_txt=Path(
                os.environ.get(
                    "VIEWER_SCHOLARLY_TXT",
                    REPO_ROOT / "tests/ocr/AlbucE_aligned_20260628_142959.txt",
                )
            ),
            filtered_kept_dir=Path(
                os.environ.get(
                    "VIEWER_FILTERED_KEPT",
                    REPO_ROOT / "data/processed/filtered_images/20260618_160948/original/kept",
                )
            ),
        )
