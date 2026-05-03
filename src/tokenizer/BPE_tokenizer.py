from pathlib import Path
import logging
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Union, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast

def setup_logger(logs_dir: Union[str, Path], run_name: str):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"{run_name}.log"

    logger = logging.getLogger("corpus")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_file}")

    return logger

def train_occitan_htr_tokenizer(input_path: Path,output_path: Path,vocab_size:int=100,logs_dir: Optional[str] = None,run_name: Optional[str] = None):
    """
    Export both:
    1. tokenizer.json
    2. Hugging Face compatible tokenizer folder
    """

    if logs_dir:
        logger = setup_logger(logs_dir, run_name)
        logger.info(f"=== Tokenizer Training Started | Run: {run_name} ===")
    else:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("tokenizer")

    corpus_files = list(input_path.glob("*.txt"))

    if not corpus_files:
        raise ValueError(f"No .txt files found in {input_path}")
    
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Number of files: {len(corpus_files)}")

    for file_path in corpus_files:
        logger.info(f" - {file_path}")

    logger.info("Tokenizer config:")
    logger.info(f" - vocab_size: {vocab_size}")
    logger.info(f" - pre_tokenizer: ByteLevel")
    logger.info(f" - decoder: ByteLevel")
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["[PAD]","[UNK]","[CLS]","[EOS]"])

    tokenizer.train([str(p) for p in corpus_files], trainer)

    vocab_size_final = len(tokenizer.get_vocab())

    logger.info("Training complete")
    logger.info(f"Final vocab size: {vocab_size_final}")

    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = TemplateProcessing(single="$A [EOS]",special_tokens=[("[EOS]", eos_id)])

    run_output_dir = output_path / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer_json_path = run_output_dir / "occitan_tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    logger.info(f"\nSaved tokenizer JSON:")
    logger.info(f" - {tokenizer_json_path}")

    
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json_path), unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", eos_token="[EOS]")
    hf_output_dir = run_output_dir / "hf_tokenizer"
    hf_tokenizer.save_pretrained(str(hf_output_dir))

    logger.info(f"\nSaved Hugging Face tokenizer folder:")
    logger.info(f" - {hf_output_dir}")

    logger.info("=== Run Summary ===")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {run_output_dir}")
    logger.info(f"Files used: {len(corpus_files)}")
    logger.info(f"Final vocab size: {vocab_size_final}")

    return hf_tokenizer
