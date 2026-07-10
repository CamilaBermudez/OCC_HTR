"""Fine-tune a Swin (image encoder) + BERT (text decoder) TrOCR-style
``VisionEncoderDecoderModel`` on the hand-annotated manuscript lines.

Architecture:
    Encoder — ``microsoft/swin-base-patch4-window7-224`` (~88M params).
    Decoder — ``bert-base-multilingual-cased`` (mBERT) with cross-attention
      injected by ``VisionEncoderDecoderModel.from_encoder_decoder_pretrained``.

Why this pairing:
- The classic TrOCR paper uses a ViT / DeiT / Swin encoder and a
  RoBERTa-style decoder. Swin gives shift-invariant attention that is
  well suited to variable-position line crops.
- BERT is encoder-only, but HuggingFace's ``VisionEncoderDecoderModel``
  wraps it into a decoder by adding cross-attention on top of its
  self-attention stack and enabling causal masking.
- mBERT's WordPiece vocab covers Latin / Romance scripts, so Old Occitan
  / Catalan tokens don't get shredded into single characters the way
  they would under an English-only BERT.

Follows the same ``scripts/`` <-> ``src/`` split as the other OCR
modules: the CLI wrapper in ``scripts/ocr/run_trocr_finetune.py`` parses
argparse, and this module hosts the logic + logging + config dump.

Data flow:
    real_folder (``<stem>.png`` + ``<stem>.gt.txt``)
      -> gather + drop empty labels
      -> stem-level train/val split (deterministic, seed-controlled)
      -> ``TrOCRLineDataset`` streams pixel_values + labels on demand
      -> ``Seq2SeqTrainer`` runs training with generation-time CER/WER
      -> best checkpoint (by lowest val CER) copied to ``best_model/``
"""

import datetime
import json
import logging
import os
import random
import subprocess
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torch.utils.data import Dataset

DEFAULT_ENCODER_ID = "microsoft/swin-base-patch4-window7-224"
DEFAULT_DECODER_ID = "bert-base-multilingual-cased"


def setup_simple_logging(
    logs_dir: str | Path, task_name: str = "trocr_finetune", run_name: str | None = None
):
    """File + console logger, same shape as the other ``src.ocr`` modules."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(logs_dir) / f"{run_name}_{task_name}.log"

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("=== %s Run Started | Run: %s ===", task_name.upper(), run_name)
    logger.info("Log file: %s", log_file)
    return logger, str(log_file)


def _get_git_commit() -> str:
    """Short git SHA at ``PROJECT_ROOT``, or ``'unknown'`` if unavailable."""
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


def _detect_device(requested: str) -> str:
    """Resolve ``'auto'`` to mps > cuda > cpu, otherwise return ``requested`` as-is."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gather_real_pairs(folder: Path, logger: logging.Logger) -> list[tuple[Path, str]]:
    """Walk ``folder`` for ``<stem>.png`` + ``<stem>.gt.txt`` pairs.

    Skips pairs where the ``.gt.txt`` is missing or empty (the annotator
    intentionally cleared a blank crop). Returns a list of
    ``(image_path, gt_text)`` tuples for downstream splitting.
    """
    pngs = sorted(folder.glob("*.png"))
    pairs: list[tuple[Path, str]] = []
    skipped_missing_gt = 0
    skipped_empty = 0
    for p in pngs:
        gt_path = p.with_suffix(".gt.txt")
        if not gt_path.is_file():
            skipped_missing_gt += 1
            continue
        text = gt_path.read_text(encoding="utf-8").strip()
        if not text:
            skipped_empty += 1
            continue
        pairs.append((p, text))
    logger.info(
        "Gathered %d (image, gt) pairs from %s (skipped %d missing gt, %d empty gt)",
        len(pairs),
        folder,
        skipped_missing_gt,
        skipped_empty,
    )
    return pairs


def _split_pairs(
    pairs: list[tuple[Path, str]],
    val_fraction: float,
    seed: int,
    logger: logging.Logger,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Deterministic train/val split at the (stem) level.

    Because the real folder has one image per stem, splitting by index is
    equivalent to splitting by stem — no augmented variants to keep
    together (unlike the ketos pipeline). We keep the "by stem" framing
    so future extensions (multiple crops per stem, augmentations) drop in
    without breaking the invariant that a variant of a train line never
    appears in val.
    """
    n = len(pairs)
    assert n >= 2, f"Need at least 2 pairs for a train/val split, got {n}"
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = set(order[:n_val])
    train = [pair for i, pair in enumerate(pairs) if i not in val_idx]
    val = [pair for i, pair in enumerate(pairs) if i in val_idx]
    assert train and val, (
        f"Split produced empty side (train={len(train)}, val={len(val)}). "
        f"Raise --val-fraction or lower the sample count."
    )
    logger.info(
        "Split: %d train / %d val (val_fraction=%.3f, seed=%d)",
        len(train),
        len(val),
        val_fraction,
        seed,
    )
    return train, val


class TrOCRLineDataset(Dataset):
    """Stream ``(pixel_values, labels)`` pairs to ``Seq2SeqTrainer``.

    The image processor rescales each PNG to the encoder's native input
    size (224x224 for Swin-base). The tokenizer pads / truncates labels
    to ``max_target_length`` and replaces pad ids with ``-100`` so that
    padded positions are ignored by the cross-entropy loss.
    """

    def __init__(
        self,
        pairs: list[tuple[Path, str]],
        image_processor,
        tokenizer,
        max_target_length: int,
    ) -> None:
        self.pairs = pairs
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self._pad_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, text = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values[0]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
        )
        labels = [tok if tok != self._pad_id else -100 for tok in encoding.input_ids]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _build_model(
    encoder_id: str,
    decoder_id: str,
    tokenizer,
    max_length: int,
    no_repeat_ngram_size: int,
    num_beams: int,
    length_penalty: float,
):
    """Build a Swin + BERT ``VisionEncoderDecoderModel`` for generation.

    ``from_encoder_decoder_pretrained`` loads both towers, initialises
    cross-attention in the decoder, and returns a wrapped model. We then
    poke the ``config`` fields that ``generate()`` and the loss both need
    to be correct (decoder_start / pad / eos ids, vocab_size, etc).
    """
    from transformers import VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_id,
        decoder_id,
        # tie_encoder_decoder=False is the default; leaving it here to make
        # the choice explicit — Swin and BERT have different hidden sizes,
        # so weight-tying between encoder embeddings and decoder embeddings
        # is not applicable.
    )
    # BERT uses [CLS] as the sequence-start marker and [SEP] as end-of-
    # sequence, so the decoder needs those wired up. Without this the
    # decoder cannot start generation and the loss will be NaN.
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # Generation-time defaults. These are stored on the config so a caller
    # doing model.generate(pixel_values) picks them up automatically at
    # inference time (matches TrOCR's original settings).
    model.config.max_length = max_length
    model.config.no_repeat_ngram_size = no_repeat_ngram_size
    model.config.num_beams = num_beams
    model.config.length_penalty = length_penalty
    model.config.early_stopping = True
    return model


def _compute_metrics_factory(tokenizer):
    """Closure over ``tokenizer`` — returns a ``compute_metrics`` fn for ``Seq2SeqTrainer``.

    Reports corpus-level CER and WER using ``rapidfuzz.distance.Levenshtein``
    (identical implementation to ``src.ocr.evaluate_ocr`` so numbers are
    directly comparable to catmus / medusa / kraken-finetune scores).
    """

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred
        # Trainer replaces label pads with -100 to mask them from the loss;
        # switch them back to the real pad id so decoding doesn't produce
        # "###" everywhere.
        label_ids = [
            [(tok if tok != -100 else tokenizer.pad_token_id) for tok in row] for row in label_ids
        ]
        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        total_char_dist = 0
        total_chars = 0
        total_word_dist = 0
        total_words = 0
        for p_text, g_text in zip(preds, labels, strict=False):
            g = g_text.strip()
            p = p_text.strip()
            if not g:
                # Empty ref — the evaluator upstream also skips these; we
                # do the same so the metric isn't dominated by pathological
                # empty-label rows.
                continue
            total_char_dist += Levenshtein.distance(p, g)
            total_chars += len(g)
            p_toks = p.split()
            g_toks = g.split()
            total_word_dist += Levenshtein.distance(p_toks, g_toks)
            total_words += len(g_toks) if g_toks else 1

        cer = total_char_dist / total_chars if total_chars else 0.0
        wer = total_word_dist / total_words if total_words else 0.0
        return {
            "cer": cer,
            "wer": wer,
            "char_acc": max(0.0, 1.0 - cer),
            "word_acc": max(0.0, 1.0 - wer),
        }

    return compute_metrics


def finetune_trocr(
    real_folder: str | Path,
    output_base_dir: str | Path,
    *,
    encoder_id: str = DEFAULT_ENCODER_ID,
    decoder_id: str = DEFAULT_DECODER_ID,
    val_fraction: float = 0.2,
    seed: int = 42,
    epochs: int = 20,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    eval_batch_size: int = 8,
    max_target_length: int = 128,
    no_repeat_ngram_size: int = 3,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    early_stopping_patience: int = 5,
    dataloader_num_workers: int = 0,
    device: str = "auto",
    logs_dir: str | Path | None = None,
    task_name: str = "trocr_finetune",
    log_config: bool = True,
) -> dict:
    """End-to-end fine-tune of a Swin+BERT VisionEncoderDecoderModel.

    Args:
        real_folder: Directory of ``<stem>.png`` + ``<stem>.gt.txt`` pairs.
        output_base_dir: Parent directory under which a
            ``trocr_<timestamp>/`` run directory is created.
        encoder_id: HuggingFace image encoder id.
        decoder_id: HuggingFace text decoder id.
        val_fraction: Fraction of pairs held out for validation.
        seed: RNG seed for the split.
        epochs: Max training epochs (early stopping may cut it short).
        learning_rate: Peak LR. 5e-5 is a common HF fine-tune default.
        batch_size, eval_batch_size: Per-device batch sizes.
        max_target_length: Tokenizer truncation length. 128 tokens
            comfortably covers CATMuS line lengths.
        no_repeat_ngram_size, num_beams, length_penalty: Generation-time
            defaults baked into ``model.config``.
        early_stopping_patience: Epochs without CER improvement before
            training halts.
        dataloader_num_workers: 0 avoids MPS's fork-safety issues.
        device: ``auto | mps | cuda | cpu``.
        logs_dir: Plain-text log location. Default: ``logs/trocr_finetune``.
        log_config: If True, dump run config JSON to the log.

    Returns:
        Dict with ``run_name``, ``run_dir``, ``best_model_dir``,
        ``eval_metrics``, ``log_path``.
    """
    # Heavy imports are lazy so ``--help`` on the CLI wrapper stays snappy
    # and unit tests that only touch helpers don't pay the transformers
    # import cost.
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    real_folder = Path(real_folder)
    output_base_dir = Path(output_base_dir)
    logs_dir = Path(logs_dir) if logs_dir else project_root / "logs" / task_name

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"trocr_{timestamp}"
    run_dir = output_base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_simple_logging(logs_dir, task_name, run_name)

    resolved_device = _detect_device(device)
    set_seed(seed)

    if log_config:
        config = {
            "run_name": run_name,
            "git_commit": _get_git_commit(),
            "timestamp": datetime.datetime.now().isoformat(),
            "real_folder": str(real_folder),
            "output_dir": str(run_dir),
            "encoder_id": encoder_id,
            "decoder_id": decoder_id,
            "val_fraction": val_fraction,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "max_target_length": max_target_length,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping_patience": early_stopping_patience,
            "device_requested": device,
            "device_resolved": resolved_device,
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")},
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        logger.info("Configuration: %s", json.dumps(config, indent=2))

    pairs = _gather_real_pairs(real_folder, logger)
    assert pairs, f"No usable (png, gt) pairs in {real_folder}"

    train_pairs, val_pairs = _split_pairs(pairs, val_fraction, seed, logger)

    logger.info("Loading image processor: %s", encoder_id)
    image_processor = AutoImageProcessor.from_pretrained(encoder_id)
    logger.info("Loading tokenizer: %s", decoder_id)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)

    train_ds = TrOCRLineDataset(train_pairs, image_processor, tokenizer, max_target_length)
    val_ds = TrOCRLineDataset(val_pairs, image_processor, tokenizer, max_target_length)

    logger.info("Building VisionEncoderDecoderModel (%s + %s)", encoder_id, decoder_id)
    model = _build_model(
        encoder_id=encoder_id,
        decoder_id=decoder_id,
        tokenizer=tokenizer,
        max_length=max_target_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        length_penalty=length_penalty,
    )

    # fp16 is a CUDA-only feature in Trainer. On MPS or CPU we stick to
    # fp32; on CUDA we opt into mixed precision for the speedup.
    use_fp16 = resolved_device == "cuda"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        predict_with_generate=True,
        generation_max_length=max_target_length,
        generation_num_beams=num_beams,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=use_fp16,
        logging_dir=str(run_dir / "tb_logs"),
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics_factory(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(
        "Training complete: %s",
        json.dumps({k: float(v) for k, v in train_result.metrics.items()}, default=str),
    )

    # Save the *best* checkpoint (already loaded thanks to
    # load_best_model_at_end) as a self-contained folder that
    # ``trocr_transcribe.py`` can point at.
    best_model_dir = run_dir / "best_model"
    logger.info("Saving best model + processor + tokenizer to %s", best_model_dir)
    trainer.save_model(str(best_model_dir))
    image_processor.save_pretrained(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    logger.info("Running final eval on val set...")
    eval_metrics = trainer.evaluate()
    logger.info(
        "Final val metrics: %s",
        json.dumps({k: float(v) for k, v in eval_metrics.items()}, default=str),
    )
    (run_dir / "final_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in eval_metrics.items()}, indent=2), encoding="utf-8"
    )

    logger.info("Run directory: %s", run_dir)
    logger.info("Best model:    %s", best_model_dir)
    logger.info("Run log:       %s", log_file)

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "best_model_dir": str(best_model_dir),
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
        "log_path": log_file,
    }
