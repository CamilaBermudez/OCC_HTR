"""Fine-tune the Kraken OCR model on the augmented synthetic dataset.

This module wires the augmented images + the corrected labels JSON
(produced by ``run_label_correction.py``) into the ``ketos train``
recogniser-training pipeline.

Important design points:

* Splits are **grouped by source line**: every augmented variant of one
  source line stays in the same split, so the model can never see a
  near-duplicate at validation that it already memorised at training.
* Staging is on-disk and persistent: we create ``train/`` and ``val/``
  subdirectories with symlinks to the augmented PNGs and sibling
  ``.gt.txt`` files. Symlinks keep disk usage low; the persistent layout
  means a failed ``ketos`` run can be retried without re-staging.
* ``ketos train`` is invoked via subprocess in ``path`` mode with
  explicit ``-t`` and ``-e`` file lists, ``--load`` for the base model,
  and ``--resize union`` so that characters not in the base codec are
  added rather than aborting training.
"""

import datetime
import json
import logging
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

_AUG_FILENAME_RE = re.compile(r"^(.+)_aug\d+\.png$")


def _sanitize_for_kraken(name: str) -> str:
    """Replace dots inside the stem with underscores.

    Kraken's ``parse_gt_path`` uses ``Path.with_suffix('')`` in a loop to
    strip *every* suffix from the image path before looking up the
    ``.gt.txt`` sibling, so a filename like ``RecChantC_ag.thes_l00086_aug00.png``
    gets stripped back to ``RecChantC_ag`` and the resolver looks for
    ``RecChantC_ag.gt.txt`` — which doesn't exist. Replacing the stem's
    dots with underscores keeps the lookup stable while preserving the
    file's extension (.png / .gt.txt).
    """
    if name.endswith(".gt.txt"):
        stem, suffix = name[: -len(".gt.txt")], ".gt.txt"
    else:
        p = Path(name)
        stem, suffix = p.stem, p.suffix
    return stem.replace(".", "_") + suffix


def setup_finetune_logging(logs_dir: str | Path, run_name: str):
    """File + console logger, same pattern as the other src/ scripts."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"{run_name}_finetune.log"

    logger = logging.getLogger("finetune_ocr")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for h in (file_handler, console):
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger, str(log_file)


def _get_git_commit() -> str:
    """Short git SHA at PROJECT_ROOT, or 'unknown' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.environ.get("PROJECT_ROOT", "."),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _source_stem(aug_filename: str) -> str | None:
    """Strip the ``_aug<NN>.png`` suffix to recover the source line stem."""
    m = _AUG_FILENAME_RE.match(aug_filename)
    return m.group(1) if m else None


def _link_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink from dst -> src; fall back to copy on failure.

    Some filesystems (e.g. mounted shares) don't support symlinks; a
    copy keeps the pipeline working at the cost of disk space.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def stage_finetune_data(
    augmented_folder: Path,
    labels_json: Path,
    staging_dir: Path,
    *,
    val_fraction: float,
    seed: int,
    smoke_size: int | None,
    logger: logging.Logger,
) -> tuple[Path, Path, dict]:
    """Stage augmented PNGs + .gt.txt siblings into train/val subdirs.

    Returns ``(train_list_path, val_list_path, stats)``. The list files
    contain one absolute image path per line, which is what ``ketos
    train`` expects via ``-t`` and ``-e``.
    """
    labels: dict[str, str] = json.loads(labels_json.read_text(encoding="utf-8"))
    assert labels, f"Empty labels file: {labels_json}"

    # Group augmented filenames by source line stem.
    by_source: dict[str, list[str]] = {}
    skipped_no_match = 0
    for aug_name in labels:
        stem = _source_stem(aug_name)
        if stem is None:
            skipped_no_match += 1
            continue
        by_source.setdefault(stem, []).append(aug_name)

    source_stems = sorted(by_source.keys())
    rng = random.Random(seed)
    rng.shuffle(source_stems)

    if smoke_size is not None:
        source_stems = source_stems[:smoke_size]
        logger.info(f"Smoke mode: restricting to {len(source_stems)} source lines")

    n_val = max(1, int(round(len(source_stems) * val_fraction)))
    val_stems = set(source_stems[:n_val])
    train_stems = set(source_stems[n_val:])
    assert train_stems and val_stems, (
        f"Split produced empty side (train={len(train_stems)}, val={len(val_stems)}). "
        f"Need at least 2 source lines."
    )

    train_dir = staging_dir / "train"
    val_dir = staging_dir / "val"
    for d in (train_dir, val_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    missing_image = 0
    for stem, aug_names in by_source.items():
        if stem in val_stems:
            split_dir, split_list = val_dir, val_paths
        elif stem in train_stems:
            split_dir, split_list = train_dir, train_paths
        else:
            # Source line was dropped by smoke_size — skip its variants.
            continue
        for aug_name in aug_names:
            src_img = augmented_folder / aug_name
            if not src_img.is_file():
                missing_image += 1
                continue
            safe_name = _sanitize_for_kraken(aug_name)
            dst_img = split_dir / safe_name
            dst_txt = split_dir / _sanitize_for_kraken(f"{src_img.stem}.gt.txt")
            _link_or_copy(src_img, dst_img)
            dst_txt.write_text(labels[aug_name], encoding="utf-8")
            # Use .absolute() — NOT .resolve() — so the path stays at the
            # staging symlink. .resolve() would follow the symlink to the
            # original image dir, where no .gt.txt sibling exists.
            split_list.append(dst_img.absolute())

    assert train_paths and val_paths, "No usable images staged — check augmented_folder paths."

    train_list = staging_dir / "train_files.txt"
    val_list = staging_dir / "val_files.txt"
    train_list.write_text("\n".join(str(p) for p in train_paths) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(str(p) for p in val_paths) + "\n", encoding="utf-8")

    stats = {
        "n_source_lines": len(source_stems),
        "n_train_source_lines": len(train_stems),
        "n_val_source_lines": len(val_stems),
        "n_train_images": len(train_paths),
        "n_val_images": len(val_paths),
        "skipped_no_match": skipped_no_match,
        "missing_image": missing_image,
    }
    logger.info(f"Staged data: {json.dumps(stats)}")
    return train_list, val_list, stats


def mix_in_real_samples(
    *,
    real_folder: Path,
    staging_dir: Path,
    train_list: Path,
    val_list: Path,
    real_train_frac: float,
    real_val_frac: float,
    real_replaces_synth_val: bool,
    seed: int,
    logger: logging.Logger,
) -> dict:
    """Drop real-manuscript ``<stem>.png + <stem>.gt.txt`` pairs into the
    staging train/val splits so the fine-tune sees real data alongside
    the synthetic augmentations.

    Three reasons this is its own step instead of being folded into
    ``stage_finetune_data``:
      - The real folder uses kraken's ``<stem>.gt.txt`` convention
        directly (no labels.json indirection, no ``_aug<NN>`` suffixes).
      - Real samples are not grouped by source line — one image = one
        sample — so the source-line split logic doesn't apply.
      - It is optional: omit ``--real-folder`` and the call is skipped.

    Behaviour:
      - Counts are derived from fractions of the folder contents so the
        split auto-scales as the corrected pool grows. ``n_train`` is
        ``floor(n_total * real_train_frac)`` and ``n_val`` is
        ``floor(n_total * real_val_frac)``; floor avoids the rounding
        case where ``round`` could push the sum past ``n_total``.
      - Pairs are sorted by stem then shuffled with ``seed``, so the
        first ``n_train`` go to train and the next ``n_val`` go to val.
      - When ``real_replaces_synth_val`` is true, val_list is rewritten
        to contain ONLY the real samples (the synthetic val accuracy
        was useless precisely because val was 100% synthetic).
      - When false, real val samples are appended to the synthetic val
        list.
      - Train always appends (real anchors mixed with synthetic).

    Asserts: real samples must form ``.png + .gt.txt`` pairs, fractions
    must be non-negative and sum to ``<= 1.0``.
    """
    assert 0.0 <= real_train_frac <= 1.0, f"real_train_frac out of range: {real_train_frac}"
    assert 0.0 <= real_val_frac <= 1.0, f"real_val_frac out of range: {real_val_frac}"
    assert real_train_frac + real_val_frac <= 1.0 + 1e-9, (
        f"real_train_frac + real_val_frac must be <= 1.0, got "
        f"{real_train_frac} + {real_val_frac} = {real_train_frac + real_val_frac}"
    )
    real_folder = Path(real_folder)
    assert real_folder.is_dir(), f"Real folder not found: {real_folder}"
    pngs = sorted(real_folder.glob("*.png"))
    pairs: list[tuple[Path, Path]] = []
    for p in pngs:
        gt = p.with_suffix(".gt.txt")
        if gt.is_file():
            pairs.append((p, gt))
    assert pairs, f"No <stem>.png + <stem>.gt.txt pairs in {real_folder}"

    n_total = len(pairs)
    n_real_train = int(n_total * real_train_frac)
    n_real_val = int(n_total * real_val_frac)
    assert n_real_train + n_real_val > 0, (
        f"Real fractions yield 0 train + 0 val from {n_total} pairs — "
        f"either raise the fractions or add more samples."
    )

    rng = random.Random(seed)
    rng.shuffle(pairs)
    train_pairs = pairs[:n_real_train]
    val_pairs = pairs[n_real_train : n_real_train + n_real_val]

    train_dir = staging_dir / "train"
    val_dir = staging_dir / "val"

    def _stage(pair_list: list[tuple[Path, Path]], dst_dir: Path) -> list[Path]:
        out: list[Path] = []
        for img, gt in pair_list:
            dst_img = dst_dir / _sanitize_for_kraken(img.name)
            dst_gt = dst_dir / _sanitize_for_kraken(gt.name)
            _link_or_copy(img, dst_img)
            _link_or_copy(gt, dst_gt)
            out.append(dst_img.absolute())
        return out

    real_train_paths = _stage(train_pairs, train_dir)
    real_val_paths = _stage(val_pairs, val_dir)

    # Append real train paths to the synthetic train list.
    with open(train_list, "a", encoding="utf-8") as f:
        for p in real_train_paths:
            f.write(f"{p}\n")

    if real_replaces_synth_val:
        val_list.write_text("\n".join(str(p) for p in real_val_paths) + "\n", encoding="utf-8")
    else:
        with open(val_list, "a", encoding="utf-8") as f:
            for p in real_val_paths:
                f.write(f"{p}\n")

    stats = {
        "real_folder": str(real_folder),
        "n_real_available": len(pairs),
        "real_train_frac": real_train_frac,
        "real_val_frac": real_val_frac,
        "n_real_train": n_real_train,
        "n_real_val": n_real_val,
        "real_replaces_synth_val": real_replaces_synth_val,
        "real_seed": seed,
    }
    logger.info(f"Mixed in real samples: {json.dumps(stats)}")
    return stats


def run_ketos_train(
    *,
    base_model: Path,
    train_list: Path,
    val_list: Path,
    output_prefix: Path,
    epochs: int,
    quit_strategy: str,
    lag: int,
    lrate: float,
    batch_size: int,
    resize: str,
    device: str,
    log_dir: Path,
    logger: logging.Logger,
) -> int:
    """Invoke ``ketos train`` as a subprocess, streaming output to the logger."""
    # We invoke ketos via a small launcher script that first monkey-patches
    # rich.Console.clear_live to be empty-stack-safe (see _ketos_launcher.py).
    # Without this, kraken's RichProgressBar crashes on
    # on_sanity_check_start with IndexError: pop from empty list.
    launcher = Path(__file__).resolve().parents[2] / "scripts" / "ocr" / "_ketos_launcher.py"
    cmd = [
        "uv",
        "run",
        "python",
        str(launcher),
        "-d",
        device,
        "train",
        "-f",
        "path",
        "--load",
        str(base_model),
        "-t",
        str(train_list),
        "-e",
        str(val_list),
        "-o",
        str(output_prefix),
        "-N",
        str(epochs),
        "-q",
        quit_strategy,
        "--lag",
        str(lag),
        "-r",
        str(lrate),
        "-B",
        str(batch_size),
        "--resize",
        resize,
        "--no-augment",
        "--log-dir",
        str(log_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))

    # Let ketos inherit the parent terminal (stdout/stderr=None) so Rich's
    # progress bar works. Capturing via PIPE makes stdout non-TTY, which
    # triggers a kraken bug: on_sanity_check_start -> clear_live() pops
    # from an empty live_stack. Inheriting the terminal also means the user
    # sees training progress live; our run log captures the surrounding
    # framing (config, paths, exit code), and tb_logs/ holds the metrics.
    rc = subprocess.call(cmd)
    logger.info("ketos train exit code: %d", rc)
    return rc


def _summarize_and_prune(
    run_dir: Path,
    logger: logging.Logger,
    *,
    keep_all_checkpoints: bool,
) -> None:
    """Extract per-epoch metrics from the model checkpoints, write a
    report, and optionally prune everything but the best checkpoint.

    Kraken stores the full training history in
    ``model.nn.user_metadata['metrics']`` — a list of ``[step, metrics_dict]``
    entries that includes the pretrained model's entire history followed
    by the epochs from the current fine-tuning run. We slice off the last
    ``completed_epochs`` entries to get only the run's own epochs.
    """
    from kraken.lib import models as kmodels  # local import: heavy, lazy

    model_files = sorted(
        p for p in run_dir.glob("model_*.mlmodel") if p.name != "model_best.mlmodel"
    )
    if not model_files:
        logger.warning("No model_*.mlmodel checkpoints found in %s", run_dir)
        return

    # Load the last checkpoint (most complete history) to extract metrics.
    last_model = model_files[-1]
    meta = kmodels.load_any(str(last_model)).nn.user_metadata
    all_metrics = meta.get("metrics", [])
    completed_epochs = int(meta.get("hyper_params", {}).get("completed_epochs", len(model_files)))
    if completed_epochs <= 0 or completed_epochs > len(all_metrics):
        logger.warning(
            "Unexpected completed_epochs=%s vs %d metric entries; " "skipping summary/prune.",
            completed_epochs,
            len(all_metrics),
        )
        return
    our_metrics = all_metrics[-completed_epochs:]

    epoch_stats: list[dict] = []
    for i, entry in enumerate(our_metrics):
        step, m = entry
        row = {
            "epoch": i,
            "step": int(step),
            "model": f"model_{i}.mlmodel",
        }
        for k, v in m.items():
            row[k] = float(v)
        epoch_stats.append(row)

    (run_dir / "epoch_stats.json").write_text(json.dumps(epoch_stats, indent=2), encoding="utf-8")

    header = [
        "epoch",
        "step",
        "val_accuracy",
        "val_word_accuracy",
        "train_loss_epoch",
        "train_loss_step",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in epoch_stats:
        lines.append(
            "| "
            + " | ".join(
                f"{row[k]:.4f}" if isinstance(row.get(k), float) else str(row.get(k, ""))
                for k in header
            )
            + " |"
        )
    (run_dir / "epoch_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = max(epoch_stats, key=lambda e: e.get("val_accuracy", -1.0))
    best_path = run_dir / best["model"]
    best_dest = run_dir / "model_best.mlmodel"
    shutil.copy2(best_path, best_dest)

    logger.info("Per-epoch stats:")
    for line in lines:
        logger.info("  %s", line)
    logger.info(
        "Best epoch: %d (val_accuracy=%.4f, val_word_accuracy=%.4f)",
        best["epoch"],
        best.get("val_accuracy", float("nan")),
        best.get("val_word_accuracy", float("nan")),
    )
    logger.info("Best model copied to: %s", best_dest)

    if keep_all_checkpoints:
        logger.info(
            "keep_all_checkpoints=True — leaving %d per-epoch checkpoints in place.",
            len(model_files),
        )
        return

    removed = 0
    for m in model_files:
        m.unlink()
        removed += 1
    logger.info("Pruned %d per-epoch checkpoints; kept model_best.mlmodel only.", removed)


def finetune(
    augmented_folder: str | Path,
    labels_json: str | Path,
    base_model: str | Path,
    output_base_dir: str | Path,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
    smoke: bool = False,
    smoke_size: int = 50,
    smoke_epochs: int = 2,
    epochs: int = -1,
    lag: int = 5,
    lrate: float = 1e-4,
    batch_size: int = 1,
    resize: str = "union",
    device: str = "cpu",
    keep_all_checkpoints: bool = False,
    logs_dir: str | Path | None = None,
    real_folder: str | Path | None = None,
    real_train_frac: float = 0.0,
    real_val_frac: float = 0.0,
    real_replaces_synth_val: bool = True,
) -> Path:
    """End-to-end fine-tune: stage data, run ``ketos train``, return run dir.

    Args:
        augmented_folder: Directory of augmented PNGs (``aug_<timestamp>/``).
        labels_json: Corresponding ``labels_<timestamp>/labels.json``.
        base_model: Path to the ``.mlmodel`` to fine-tune from.
        output_base_dir: Parent directory under which a
            ``finetune_<timestamp>/`` run directory is created.
        val_fraction: Fraction of *source lines* (not images) held out
            for validation. Default 0.1 (90/10 split).
        seed: RNG seed for the source-line shuffle / split.
        smoke: If True, restrict to ``smoke_size`` source lines and run
            for ``smoke_epochs`` epochs with a fixed quit strategy.
        smoke_size, smoke_epochs: Sizing knobs for smoke mode.
        epochs: Epoch budget when ``smoke=False``. ``-1`` plus
            ``quit=early`` means train until the early-stopping
            criterion fires.
        lag: Early-stopping patience (epochs without improvement).
        lrate: Learning rate. Default ``1e-4`` — an order of magnitude
            below the ``ketos`` default, which is the usual fine-tuning
            convention.
        batch_size: ``ketos -B``. Default 1 (the ``ketos`` default,
            CPU-friendly).
        resize: ``ketos --resize``. Default ``union`` so any characters
            absent from the base codec are added rather than aborting.
        device: ``ketos -d``. ``cpu`` or ``cuda:0`` etc.
        logs_dir: Optional plain-text run log location.

    Returns:
        Path to the ``finetune_<timestamp>/`` run directory.
    """
    augmented_folder = Path(augmented_folder)
    labels_json = Path(labels_json)
    base_model = Path(base_model)
    output_base_dir = Path(output_base_dir)

    assert augmented_folder.is_dir(), f"Augmented folder not found: {augmented_folder}"
    assert labels_json.is_file(), f"Labels JSON not found: {labels_json}"
    assert base_model.is_file(), f"Base model not found: {base_model}"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"finetune_{timestamp}"
    run_dir = output_base_dir / run_name
    staging_dir = run_dir / "data"
    tb_log_dir = run_dir / "tb_logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    if logs_dir:
        logger, log_file = setup_finetune_logging(logs_dir, run_name)
    else:
        logger = logging.getLogger("finetune_ocr")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        log_file = None

    logger.info(f"=== Fine-tune started | Run: {run_name} ===")

    effective_epochs = smoke_epochs if smoke else epochs
    # Smoke runs now use the early-stopping quit strategy too, so a long
    # smoke (e.g. SMOKE_EPOCHS=15) can terminate as soon as val_accuracy
    # plateaus for `lag` epochs instead of always burning the full budget.
    # smoke_epochs becomes the MAX-epoch cap rather than a fixed count.
    # On very short smokes (SMOKE_EPOCHS <= lag) early stopping never
    # fires, so behaviour is identical to the old "fixed" mode.
    effective_quit = "early"
    effective_smoke_size = smoke_size if smoke else None

    config = {
        "run": run_name,
        "git": _get_git_commit(),
        "augmented_folder": str(augmented_folder),
        "labels_json": str(labels_json),
        "base_model": str(base_model),
        "output_dir": str(run_dir),
        "val_fraction": val_fraction,
        "seed": seed,
        "smoke": smoke,
        "smoke_size": smoke_size,
        "effective_epochs": effective_epochs,
        "effective_quit": effective_quit,
        "lag": lag,
        "lrate": lrate,
        "batch_size": batch_size,
        "resize": resize,
        "device": device,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info(f"Config: {json.dumps(config)}")

    train_list, val_list, stats = stage_finetune_data(
        augmented_folder=augmented_folder,
        labels_json=labels_json,
        staging_dir=staging_dir,
        val_fraction=val_fraction,
        seed=seed,
        smoke_size=effective_smoke_size,
        logger=logger,
    )
    if real_folder is not None and (real_train_frac > 0 or real_val_frac > 0):
        real_stats = mix_in_real_samples(
            real_folder=Path(real_folder),
            staging_dir=staging_dir,
            train_list=train_list,
            val_list=val_list,
            real_train_frac=real_train_frac,
            real_val_frac=real_val_frac,
            real_replaces_synth_val=real_replaces_synth_val,
            seed=seed,
            logger=logger,
        )
        stats.update(real_stats)
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    rc = run_ketos_train(
        base_model=base_model,
        train_list=train_list,
        val_list=val_list,
        output_prefix=run_dir / "model",
        epochs=effective_epochs,
        quit_strategy=effective_quit,
        lag=lag,
        lrate=lrate,
        batch_size=batch_size,
        resize=resize,
        device=device,
        log_dir=tb_log_dir,
        logger=logger,
    )

    assert rc == 0, f"ketos train failed with exit code {rc} — see {log_file}"

    saved = sorted(run_dir.glob("model_*.mlmodel"))
    logger.info(f"Saved checkpoints: {[p.name for p in saved]}")

    _summarize_and_prune(run_dir, logger, keep_all_checkpoints=keep_all_checkpoints)

    if log_file:
        logger.info(f"Run log (text): {log_file}")

    return run_dir
