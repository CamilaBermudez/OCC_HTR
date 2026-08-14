"""Fine-tune a TrOCR-style ``VisionEncoderDecoderModel`` on the
hand-annotated manuscript lines.

Two modes, controlled by the CLI:

1. **Pretrained TrOCR** (``--pretrained-model-id``): load a fully-
   assembled checkpoint like ``microsoft/trocr-base-handwritten``
   (ViT+RoBERTa) whose cross-attention is already pre-trained on 34M
   synthetic + IAM handwriting lines. This is the recommended path.

2. **From-scratch encoder+decoder** (``--encoder-id`` +
   ``--decoder-id``): build a fresh model via
   ``VisionEncoderDecoderModel.from_encoder_decoder_pretrained``,
   defaulting to Swin-base + mBERT. Kept for ablation. **Note**: the
   randomly-initialised cross-attention layers here (~57M params)
   were not trainable to competitive quality on 600 real + 5000 aug
   pairs — see ``spec.md`` §6.3.

Follows the same ``scripts/`` <-> ``src/`` split as the other OCR
modules: the CLI wrapper in ``scripts/ocr/run_trocr_finetune.py`` parses
argparse, and this module hosts the logic + logging + config dump.

Data flow:
    real_folder (``<stem>.png`` + ``<stem>.gt.txt``)
      + optional augmented_folder (``<stem>_aug<NN>.png``) + labels.json
      -> gather + drop empty labels
      -> split by SOURCE stem: every augmented variant of a given source
         line stays in the same split as the real image, so the model
         never sees a near-duplicate at val that it also trained on
      -> ``TrOCRLineDataset`` streams pixel_values + labels on demand
      -> ``Seq2SeqTrainer`` runs training with generation-time CER/WER
      -> best checkpoint (by lowest val CER) copied to ``best_model/``
"""

import datetime
import json
import logging
import os
import random
import re
import subprocess
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torch.utils.data import Dataset

from src.ocr.image_prep import DEFAULT_RESIZE_MODE, prepare_image

DEFAULT_ENCODER_ID = "microsoft/swin-base-patch4-window7-224"
DEFAULT_DECODER_ID = "bert-base-multilingual-cased"

# The augmentation pipeline names its outputs
# ``<src_stem>_aug<NN>.png`` for external-corpus renders and
# ``<annotated_stem>.gt_l<NN>_aug<NN>.png`` when the source text came
# from a real annotated line's ``.gt.txt`` (each ``.gt_l<NN>`` = one
# rendering pass over that text with a different font/glyph mix).
#
# The optional ``\.gt_l\d+`` group in the regex strips that inner render
# index so both variants collapse to the SAME source stem as the real
# image on disk — otherwise the source-stem split treats a real image
# and its synthetic re-render as independent, and the same underlying
# text can end up on opposite sides of the train/val cut. Same
# invariant the kraken fine-tune's ``stage_finetune_data`` is supposed
# to enforce (kraken uses the older greedy regex and has the same leak
# — should port the same fix over when that pipeline is next touched).
_AUG_FILENAME_RE = re.compile(r"^(.+?)(?:\.gt_l\d+)?_aug\d+\.png$")


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


def _gather_real_pairs(folder: Path, logger: logging.Logger) -> list[tuple[Path, str, str]]:
    """Walk ``folder`` for ``<stem>.png`` + ``<stem>.gt.txt`` pairs.

    Skips pairs where the ``.gt.txt`` is missing or empty (the annotator
    intentionally cleared a blank crop). Returns a list of
    ``(image_path, gt_text, source_stem)`` triples. For real images the
    source stem is just the image stem — the aug pool uses the same
    field to carry the pre-augmentation stem, so downstream code can
    treat both pools uniformly.
    """
    pngs = sorted(folder.glob("*.png"))
    triples: list[tuple[Path, str, str]] = []
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
        triples.append((p, text, p.stem))
    logger.info(
        "Gathered %d real triples from %s (skipped %d missing gt, %d empty gt)",
        len(triples),
        folder,
        skipped_missing_gt,
        skipped_empty,
    )
    return triples


def _gather_augmented_pairs(
    augmented_folder: Path,
    labels_json: Path,
    logger: logging.Logger,
) -> list[tuple[Path, str, str]]:
    """Load augmented ``<stem>_aug<NN>.png`` + labels.json into triples.

    The augmentation pipeline emits filenames like
    ``72_f_067v_068_line_148_aug03.png`` and stores their labels in a
    JSON map. We strip the ``_aug<NN>`` suffix to recover the source
    stem — that's the key we split on so all N variants of a line stay
    on the same side of the train/val cut.
    """
    labels: dict[str, str] = json.loads(labels_json.read_text(encoding="utf-8"))
    triples: list[tuple[Path, str, str]] = []
    skipped_no_suffix = 0
    skipped_missing_img = 0
    skipped_empty_text = 0
    for aug_name, raw_text in labels.items():
        m = _AUG_FILENAME_RE.match(aug_name)
        if m is None:
            skipped_no_suffix += 1
            continue
        source_stem = m.group(1)
        img_path = augmented_folder / aug_name
        if not img_path.is_file():
            skipped_missing_img += 1
            continue
        text = (raw_text or "").strip()
        if not text:
            skipped_empty_text += 1
            continue
        triples.append((img_path, text, source_stem))
    logger.info(
        "Gathered %d aug triples from %s (skipped %d no _aug<NN> suffix, "
        "%d missing image, %d empty text)",
        len(triples),
        augmented_folder,
        skipped_no_suffix,
        skipped_missing_img,
        skipped_empty_text,
    )
    return triples


def _split_by_source_stem(
    triples: list[tuple[Path, str, str]],
    val_fraction: float,
    seed: int,
    logger: logging.Logger,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Split source stems into train/val, then distribute all their triples.

    Grouping by source stem is the invariant that makes the val CER
    trustworthy: if source line X has 1 real image + 5 augmentations,
    all 6 pairs land on the same side of the split. Otherwise a val
    row would be a near-duplicate of a train row and the metric would
    silently overstate real generalisation.
    """
    by_stem: dict[str, list[tuple[Path, str]]] = {}
    for path, text, source_stem in triples:
        by_stem.setdefault(source_stem, []).append((path, text))
    stems = sorted(by_stem.keys())
    assert len(stems) >= 2, f"Need at least 2 source stems for a train/val split, got {len(stems)}"
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val_stems = max(1, int(round(len(stems) * val_fraction)))
    val_stems = set(stems[:n_val_stems])

    train_pairs: list[tuple[Path, str]] = []
    val_pairs: list[tuple[Path, str]] = []
    for stem in stems:
        target = val_pairs if stem in val_stems else train_pairs
        target.extend(by_stem[stem])

    assert train_pairs and val_pairs, (
        f"Split produced empty side (train={len(train_pairs)}, val={len(val_pairs)}). "
        f"Raise --val-fraction or lower the sample count."
    )
    logger.info(
        "Split by source stem: %d train pairs (%d stems) / %d val pairs (%d stems), "
        "val_fraction=%.3f, seed=%d",
        len(train_pairs),
        len(stems) - n_val_stems,
        len(val_pairs),
        n_val_stems,
        val_fraction,
        seed,
    )
    return train_pairs, val_pairs


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
        resize_mode: str = DEFAULT_RESIZE_MODE,
    ) -> None:
        self.pairs = pairs
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.resize_mode = resize_mode
        self._pad_id = tokenizer.pad_token_id
        # Every label sequence must end with eos so the decoder learns to
        # stop. BERT/RoBERTa tokenizers append [SEP]/</s> on their own; GPT-2
        # does not — for decoders like it we append eos manually in
        # __getitem__ (otherwise the model over-generates at inference).
        self._eos_id = tokenizer.eos_token_id
        probe = tokenizer("probe").input_ids
        _adds_eos = bool(probe) and self._eos_id is not None and probe[-1] == self._eos_id
        self._will_append_eos = (not _adds_eos) and (self._eos_id is not None)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, text = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        image = prepare_image(image, self.image_processor, self.resize_mode)
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values[0]
        max_len = self.max_target_length
        # Reserve a slot for the manually-appended eos when the tokenizer
        # doesn't add one, so the sequence still ends with eos after truncation.
        budget = max_len - 1 if self._will_append_eos else max_len
        ids = self.tokenizer(text, truncation=True, max_length=budget).input_ids
        if self._will_append_eos:
            ids = ids + [self._eos_id]
        # Right-pad to max_len; pad positions are masked to -100 in the loss.
        # pad is a DISTINCT token from eos, so eos itself is never masked.
        if len(ids) < max_len:
            ids = ids + [self._pad_id] * (max_len - len(ids))
        labels = [tok if tok != self._pad_id else -100 for tok in ids]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_custom_tokenizer(tokenizer_dir: Path, logger: logging.Logger):
    """Load a custom HF tokenizer folder and re-attach its Metaspace decoder.

    ``src/tokenizer/BPE_tokenizer.py`` builds a char-level BPE with a Metaspace
    pre-tokenizer + decoder, but the Metaspace **decoder** serialises to
    ``null`` in tokenizer.json (a tokenizers quirk). Loading it back therefore
    gives a tokenizer that encodes correctly but decodes with spurious spaces
    between every token. We re-attach ``Metaspace()`` on the backend tokenizer
    so ``decode`` round-trips exactly (verified: 0 CER on the medieval samples).
    """
    from tokenizers.decoders import Metaspace as _MetaspaceDecoder
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    tok.backend_tokenizer.decoder = _MetaspaceDecoder()
    logger.info(
        "Loaded custom tokenizer from %s (vocab=%d, re-attached Metaspace decoder)",
        tokenizer_dir,
        len(tok),
    )
    return tok


def _warm_start_embeddings(
    new_emb, old_emb, tokenizer, pretrained_model_id: str, logger: logging.Logger
) -> int:
    """Copy pretrained rows for custom tokens whose surface text matches a
    single pretrained token (a safe head-start; see Q&A on the custom-BPE run).

    For each custom token, recover its surface string (Metaspace ``▁`` -> space),
    encode it with the pretrained tokenizer, and if it maps to exactly ONE
    pretrained token, copy that token's embedding into the new matrix. Single
    Latin chars and space-prefixed variants match; multi-byte medieval glyphs
    (which the byte-level BPE splits into >1 token) and most merges don't, and
    keep their random init. Best-effort: any failure leaves random init intact.
    Returns the number of rows warm-started.
    """
    from transformers import AutoTokenizer

    try:
        src_tok = AutoTokenizer.from_pretrained(pretrained_model_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("Warm-start skipped (could not load %s: %s)", pretrained_model_id, e)
        return 0
    specials = set(tokenizer.all_special_ids)
    n = 0
    for i in range(len(tokenizer)):
        if i in specials:
            continue
        piece = tokenizer.convert_ids_to_tokens(i)
        if piece is None:
            continue
        surface = piece.replace("▁", " ")  # Metaspace marker -> space
        if not surface:
            continue
        src_ids = src_tok(surface, add_special_tokens=False).input_ids
        if len(src_ids) == 1 and src_ids[0] < old_emb.shape[0]:
            new_emb[i] = old_emb[src_ids[0]]
            n += 1
    logger.info(
        "Warm-started %d/%d custom-vocab embeddings from %s (surface-string match)",
        n,
        len(tokenizer),
        pretrained_model_id,
    )
    return n


def _reinit_vocab_layers(
    model, tokenizer, pretrained_model_id: str | None, logger: logging.Logger
) -> None:
    """Swap a pretrained decoder onto a new (smaller) custom vocabulary.

    Only the **vocab-tied** layers are reset: the token-embedding matrix and
    the LM head are resized to ``len(tokenizer)`` and re-initialised from the
    decoder's own init distribution. The custom tokenizer's ids do NOT
    correspond to the pretrained tokenizer's, so there is no per-token mapping
    to preserve — every row starts fresh, then a surface-string warm-start
    (:func:`_warm_start_embeddings`) copies pretrained rows for the tokens that
    DO have an identical single-token surface (mostly the Latin chars).
    Everything else (the whole ViT encoder, and the decoder's self-attention,
    cross-attention, FFN, layernorms) keeps its pretrained weights. spec §6.5.22.
    """
    import torch.nn as nn

    new_vocab = len(tokenizer)
    old_emb = model.decoder.get_input_embeddings().weight.data.clone()
    model.decoder.resize_token_embeddings(new_vocab)
    std = getattr(model.decoder.config, "initializer_range", 0.02)
    emb = model.decoder.get_input_embeddings()
    nn.init.normal_(emb.weight, mean=0.0, std=std)
    if pretrained_model_id is not None:
        _warm_start_embeddings(emb.weight.data, old_emb, tokenizer, pretrained_model_id, logger)
    out = model.decoder.get_output_embeddings()
    if out is not None and out.weight is not emb.weight:  # untied LM head
        nn.init.normal_(out.weight, mean=0.0, std=std)
    if out is not None and getattr(out, "bias", None) is not None:
        nn.init.zeros_(out.bias)
    model.config.vocab_size = new_vocab
    model.config.decoder.vocab_size = new_vocab
    # decoder_start / pad / eos must come from the NEW tokenizer, overriding the
    # pretrained checkpoint's ids. Custom BPE specials: [PAD] [UNK] [CLS] [EOS];
    # [CLS] is the decoder-start (bos), [EOS] the stop token.
    dec_start = tokenizer.cls_token_id
    if dec_start is None:
        dec_start = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    assert dec_start is not None and eos is not None and pad is not None, (
        "Custom tokenizer must expose cls/bos, eos and pad tokens; got "
        f"cls={dec_start} eos={eos} pad={pad}"
    )
    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id = dec_start
        cfg.pad_token_id = pad
        cfg.eos_token_id = eos
    logger.info(
        "Re-initialised vocab-tied layers to custom vocab=%d "
        "(decoder_start=%d pad=%d eos=%d); pretrained encoder + decoder body kept",
        new_vocab,
        dec_start,
        pad,
        eos,
    )


def _build_model(
    tokenizer,
    max_length: int,
    no_repeat_ngram_size: int,
    num_beams: int,
    length_penalty: float,
    *,
    pretrained_model_id: str | None = None,
    encoder_id: str | None = None,
    decoder_id: str | None = None,
    reinit_vocab: bool = False,
):
    """Build a ``VisionEncoderDecoderModel`` for generation.

    Two modes:

    - ``pretrained_model_id`` set — load a fully-assembled TrOCR
      checkpoint (e.g. ``microsoft/trocr-base-handwritten``). The
      cross-attention is already trained on 34M synthetic + IAM
      handwriting lines; we only need to override generation-time
      defaults so beam-search matches the caller's config.
    - Otherwise — build the model from a vision encoder + text
      decoder via ``from_encoder_decoder_pretrained``. Cross-attention
      layers are freshly initialised — this is the mode that struggled
      on 600 real lines (see ``spec.md`` §6.3), included for
      completeness / ablation.

    Special-token ids (``decoder_start``, ``pad``, ``eos``) need to
    match whichever tokenizer the caller loaded. For pretrained TrOCR,
    the checkpoint already carries the right ids; we still mirror them
    onto ``generation_config`` for defence.
    """
    from transformers import AutoConfig, VisionEncoderDecoderModel

    if pretrained_model_id is not None:
        model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_id)

        # microsoft/trocr-base-handwritten stores generation defaults
        # on model.generation_config, but ``Trainer.compute_loss()``
        # reads ``pad_token_id`` and ``decoder_start_token_id`` from
        # ``model.config`` during teacher-forced label shifting.
        # Without them set on model.config the forward pass raises
        # ``AttributeError: 'VisionEncoderDecoderConfig' object has no
        # attribute 'pad_token_id'``. Mirror the values onto both
        # objects, falling back to the tokenizer if generation_config
        # doesn't carry them. TrOCR convention: decoder_start = eos.
        def _resolve(gc_attr: str, fallback):
            v = getattr(model.generation_config, gc_attr, None)
            return v if v is not None else fallback

        pad = _resolve("pad_token_id", tokenizer.pad_token_id)
        eos = _resolve(
            "eos_token_id",
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.sep_token_id,
        )
        dec_start = _resolve("decoder_start_token_id", eos)
        for cfg in (model.config, model.generation_config):
            cfg.decoder_start_token_id = dec_start
            cfg.pad_token_id = pad
            cfg.eos_token_id = eos

        # Custom-vocab experiment: replace the vocab-tied layers (embeddings +
        # LM head) so the pretrained ViT+RoBERTa decoder is retargeted onto the
        # custom char-BPE. This OVERRIDES the special-id block just above with
        # the custom tokenizer's ids. spec §6.5.22.
        if reinit_vocab:
            _reinit_vocab_layers(
                model, tokenizer, pretrained_model_id, logging.getLogger("trocr_finetune")
            )
    else:
        assert (
            encoder_id is not None and decoder_id is not None
        ), "Must set either pretrained_model_id or (encoder_id, decoder_id)"
        # Force the decoder into decoder-mode with cross-attention by passing
        # an explicit decoder config. BERT / RoBERTa get these flags set
        # implicitly by from_encoder_decoder_pretrained, but GPT-2's config
        # (transformers 5.x) does not declare ``is_decoder`` — and passing
        # ``decoder_is_decoder=True`` as a kwarg does NOT help because
        # ``AutoConfig.from_pretrained(..., return_unused_kwargs=True)``
        # silently drops kwargs the config doesn't declare. Building the config
        # ourselves and setting the attributes via setattr sidesteps that
        # filtering, and passing ``decoder_config=`` bypasses the internal
        # ``AutoConfig.from_pretrained`` path that trips the AttributeError.
        decoder_config = AutoConfig.from_pretrained(decoder_id)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_id,
            decoder_id,
            decoder_config=decoder_config,
            # tie_encoder_decoder=False is the default; Swin/ViT and the text
            # decoder have different hidden sizes, so weight-tying is N/A.
        )
        # Grow the decoder embedding matrix to cover any special tokens added
        # to the tokenizer (e.g. the distinct [PAD] added for GPT-2). No-op
        # when the tokenizer vocab already matches the decoder's.
        model.decoder.resize_token_embeddings(len(tokenizer))
        # Wire up the decoder's sequence-start / eos / pad tokens. BERT uses
        # [CLS]/[SEP], RoBERTa uses <s>/</s> (both exposed as cls/sep by their
        # tokenizers), but GPT-2 has neither — only bos/eos and no pad. Resolve
        # each with fallbacks so any decoder (BERT, RoBERTa, GPT-2) works.
        # Without a valid decoder_start_token_id the decoder cannot start
        # generation and the loss is NaN. STRUCTURAL config fields.
        start_id = tokenizer.cls_token_id
        if start_id is None:
            start_id = tokenizer.bos_token_id
        if start_id is None:
            start_id = tokenizer.eos_token_id
        eos_id = tokenizer.sep_token_id
        if eos_id is None:
            eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        assert (
            start_id is not None and eos_id is not None
        ), f"Decoder {decoder_id!r} tokenizer exposes no usable start/eos token"
        model.config.decoder_start_token_id = start_id
        model.config.pad_token_id = pad_id
        model.config.eos_token_id = eos_id
        model.config.vocab_size = len(tokenizer)
        model.config.decoder.vocab_size = len(tokenizer)
        # Generation-time defaults. transformers 5.x refuses to read these
        # from ``model.config`` any more (raises ValueError on generate()) —
        # they must live on ``model.generation_config``.
        model.generation_config.decoder_start_token_id = start_id
        model.generation_config.pad_token_id = pad_id
        model.generation_config.eos_token_id = eos_id

    # Generation defaults common to both modes.
    model.generation_config.max_length = max_length
    model.generation_config.no_repeat_ngram_size = no_repeat_ngram_size
    model.generation_config.num_beams = num_beams
    model.generation_config.length_penalty = length_penalty
    model.generation_config.early_stopping = True
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
    real_folder: str | Path | None,
    output_base_dir: str | Path,
    *,
    augmented_folder: str | Path | None = None,
    labels_json: str | Path | None = None,
    max_aug_samples: int | None = None,
    pretrained_model_id: str | None = None,
    encoder_id: str = DEFAULT_ENCODER_ID,
    decoder_id: str = DEFAULT_DECODER_ID,
    val_fraction: float = 0.2,
    seed: int = 42,
    epochs: int = 20,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    max_target_length: int = 128,
    no_repeat_ngram_size: int = 3,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    early_stopping_patience: int = 5,
    dataloader_num_workers: int = 0,
    device: str = "auto",
    resize_mode: str = DEFAULT_RESIZE_MODE,
    custom_tokenizer: str | Path | None = None,
    logs_dir: str | Path | None = None,
    task_name: str = "trocr_finetune",
    log_config: bool = True,
) -> dict:
    """End-to-end fine-tune of a Swin+BERT VisionEncoderDecoderModel.

    Args:
        real_folder: Directory of ``<stem>.png`` + ``<stem>.gt.txt`` pairs.
        output_base_dir: Parent directory under which a
            ``trocr_<timestamp>/`` run directory is created.
        augmented_folder: Optional directory of augmented PNGs named
            ``<src_stem>_aug<NN>.png``. When set with ``labels_json``,
            these get mixed into the training pool and split alongside
            the real images at the source-stem level so aug variants
            of a val line never appear in train.
        labels_json: Optional ``labels.json`` mapping augmented filenames
            to their (corrected) text. Required if ``augmented_folder``
            is set.
        max_aug_samples: Optional cap on the number of augmented pairs
            after gathering. If the pool has more, a deterministic
            seed-controlled random subsample of this size is kept.
            ``None`` = use every augmented pair (fine on GPU; on MPS the
            full 266k kraken pool would take days per epoch).
        pretrained_model_id: If set, load a fully-assembled TrOCR
            checkpoint via ``VisionEncoderDecoderModel.from_pretrained``
            (e.g. ``microsoft/trocr-base-handwritten``). Cross-attention
            is already trained; the image processor and tokenizer are
            loaded from the same checkpoint so ``encoder_id`` and
            ``decoder_id`` are ignored. Recommended path — the
            from-scratch Swin+BERT builds under the other branch
            struggle on this data scale (§6.3 in spec.md).
        encoder_id: HuggingFace image encoder id.
        decoder_id: HuggingFace text decoder id.
        val_fraction: Fraction of source stems held out for validation.
        seed: RNG seed for the split.
        epochs: Max training epochs (early stopping may cut it short).
        learning_rate: Peak LR. 5e-5 is a common HF fine-tune default.
        batch_size, eval_batch_size: Per-device batch sizes.
        max_target_length: Tokenizer truncation length. 128 tokens
            comfortably covers CATMuS line lengths.
        no_repeat_ngram_size, num_beams, length_penalty: Generation-time
            defaults baked into ``model.generation_config``.
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
    real_folder = Path(real_folder) if real_folder else None
    output_base_dir = Path(output_base_dir)
    augmented_folder = Path(augmented_folder) if augmented_folder else None
    labels_json = Path(labels_json) if labels_json else None
    custom_tokenizer = Path(custom_tokenizer) if custom_tokenizer else None
    assert custom_tokenizer is None or pretrained_model_id is not None, (
        "custom_tokenizer is only supported with pretrained_model_id (it retargets "
        "the pretrained decoder's vocab); the from-scratch branch already accepts "
        "any decoder tokenizer via decoder_id."
    )
    use_augmented = augmented_folder is not None and labels_json is not None
    assert not (augmented_folder is None) ^ (labels_json is None), (
        "augmented_folder and labels_json must be provided together, "
        f"got augmented_folder={augmented_folder} labels_json={labels_json}"
    )
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
            "augmented_folder": str(augmented_folder) if augmented_folder else None,
            "labels_json": str(labels_json) if labels_json else None,
            "use_augmented": use_augmented,
            "max_aug_samples": max_aug_samples,
            "pretrained_model_id": pretrained_model_id,
            "custom_tokenizer": str(custom_tokenizer) if custom_tokenizer else None,
            "model_source": (
                "pretrained_trocr" if pretrained_model_id else "encoder_decoder_pretrained"
            ),
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

    # real_folder is optional: omit it for a synthetic-only Stage-1 pretrain
    # (spec §6.13/§6.5.26 clean two-stage). At least one source must be present.
    real_triples = _gather_real_pairs(real_folder, logger) if real_folder else []
    assert (
        real_triples or use_augmented
    ), "Need at least one training source: real_folder and/or augmented_folder+labels_json"

    if use_augmented and not real_triples:
        aug_triples = _gather_augmented_pairs(augmented_folder, labels_json, logger)
        assert aug_triples, f"augmented_folder yielded 0 usable pairs ({augmented_folder})"
        if max_aug_samples is not None and len(aug_triples) > max_aug_samples:
            aug_rng = random.Random(seed)
            aug_triples = aug_rng.sample(aug_triples, max_aug_samples)
            logger.info("Subsampled augmented pool to %d pairs (seed=%d)", max_aug_samples, seed)
        all_triples = aug_triples
        logger.info(
            "Synthetic-only pool: %d aug pairs across %d source stems (no real lines)",
            len(all_triples),
            len({t[2] for t in all_triples}),
        )
    elif use_augmented:
        aug_triples = _gather_augmented_pairs(augmented_folder, labels_json, logger)
        assert aug_triples, (
            f"augmented_folder + labels_json set but yielded 0 usable pairs "
            f"({augmented_folder}, {labels_json})"
        )
        if max_aug_samples is not None and len(aug_triples) > max_aug_samples:
            # Deterministic subsample so re-running with the same seed
            # trains on the same aug subset. We use a separate Random
            # instance rather than the module-level ``random`` so
            # transformer's ``set_seed(seed)`` doesn't perturb this pick.
            aug_rng = random.Random(seed)
            aug_triples = aug_rng.sample(aug_triples, max_aug_samples)
            logger.info("Subsampled augmented pool to %d pairs (seed=%d)", max_aug_samples, seed)
        all_triples = real_triples + aug_triples
        n_source_stems = len({t[2] for t in all_triples})
        logger.info(
            "Combined pool: %d pairs across %d source stems (%d real + %d aug)",
            len(all_triples),
            n_source_stems,
            len(real_triples),
            len(aug_triples),
        )
    else:
        all_triples = real_triples

    train_pairs, val_pairs = _split_by_source_stem(all_triples, val_fraction, seed, logger)

    # Loader source depends on the mode. Pretrained TrOCR bundles image
    # processor + tokenizer + model into the same checkpoint; the
    # encoder-decoder-pretrained path loads them separately.
    if pretrained_model_id is not None:
        image_processor = AutoImageProcessor.from_pretrained(pretrained_model_id)
        if custom_tokenizer is not None:
            logger.info("Loading image processor: %s", pretrained_model_id)
            tokenizer = _load_custom_tokenizer(custom_tokenizer, logger)
        else:
            logger.info("Loading image processor + tokenizer: %s", pretrained_model_id)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
    else:
        logger.info("Loading image processor: %s", encoder_id)
        image_processor = AutoImageProcessor.from_pretrained(encoder_id)
        logger.info("Loading tokenizer: %s", decoder_id)
        tokenizer = AutoTokenizer.from_pretrained(decoder_id)
        # Some causal decoders (e.g. GPT-2) ship without a pad token. Add a
        # DISTINCT [PAD] rather than reusing eos: if pad == eos, masking the
        # pad positions in the labels (-100) also masks eos, so the model
        # never learns to emit eos and over-generates unboundedly at inference
        # (observed GPT-2 failure: CER ~6, run-on output). A separate [PAD]
        # keeps eos a real, learnable target. Embeddings are resized to cover
        # the new token in _build_model. No-op for BERT/RoBERTa (already have
        # a distinct pad).
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("Decoder tokenizer had no pad token; added a distinct [PAD] token")

    train_ds = TrOCRLineDataset(
        train_pairs, image_processor, tokenizer, max_target_length, resize_mode=resize_mode
    )
    val_ds = TrOCRLineDataset(
        val_pairs, image_processor, tokenizer, max_target_length, resize_mode=resize_mode
    )

    if pretrained_model_id is not None:
        logger.info("Building VisionEncoderDecoderModel (pretrained: %s)", pretrained_model_id)
    else:
        logger.info("Building VisionEncoderDecoderModel (%s + %s)", encoder_id, decoder_id)
    model = _build_model(
        tokenizer=tokenizer,
        max_length=max_target_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        length_penalty=length_penalty,
        pretrained_model_id=pretrained_model_id,
        encoder_id=None if pretrained_model_id else encoder_id,
        decoder_id=None if pretrained_model_id else decoder_id,
        reinit_vocab=custom_tokenizer is not None,
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=use_fp16,
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        # ``report_to=[]`` disables all integrations (tensorboard, wandb,
        # comet, etc). Enabling any of them adds a hard dep to
        # pyproject.toml; the plain-text run log + ``config.json`` +
        # ``final_metrics.json`` already cover our needs. Flip to
        # ``["tensorboard"]`` if you add tensorboard to deps later.
        report_to=[],
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
    if custom_tokenizer is not None:
        # save_pretrained serialises the Metaspace decoder as null (tokenizers
        # quirk), so a plain reload would decode with spurious inter-token
        # spaces. Inject the decoder spec into the saved tokenizer.json so the
        # model is self-contained — plain AutoTokenizer.from_pretrained then
        # round-trips correctly with no re-attach needed downstream.
        tok_json = best_model_dir / "tokenizer.json"
        spec = json.loads(tok_json.read_text(encoding="utf-8"))
        spec["decoder"] = {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        }
        tok_json.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        logger.info("Injected Metaspace decoder into %s (self-contained round-trip)", tok_json)
    # Persist the line-resize mode (pad/stretch) so transcription reproduces
    # the exact preprocessing this model was trained with (train/inference
    # MUST match, or accuracy silently collapses).
    (best_model_dir / "resize_mode.txt").write_text(resize_mode, encoding="utf-8")

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
