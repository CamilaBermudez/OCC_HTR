#!/bin/bash
# Sequential driver for the TrOCR + staged Swin+BERT phases that failed on
# 2026-07-22 due to system python3 missing rapidfuzz/transformers/torch.
# Fix: prefix every call with `uv run` so it runs in the uv-managed .venv.
#
# Phases (in order, single L4 GPU, strictly sequential):
#   1. TrOCR ViT+RoBERTa + Dataset A'' (matched_cometa)
#   2. TrOCR ViT+RoBERTa + Dataset B'' (medical)
#   3. TrOCR Swin+BERT single-stage + Dataset A''
#   4. TrOCR Swin+BERT single-stage + Dataset B''
#   5. Swin+BERT Stage 1a — 30k COMETA pretrain
#   6. Swin+BERT Stage 2a — fine-tune Stage 1a on A''
#   7. Swin+BERT Stage 2b — fine-tune Stage 1a on B''
#
# Assumes Kraken Run 1 + Run 2 are ALREADY done (skip them).
# If a phase fails the driver logs the exit code and continues to the next.
#
# Usage on the VM (from JupyterLab terminal):
#   cd /home/jupyter/OCC_HTR
#   nohup bash scripts/ocr/queue_trocr_and_staged.sh > logs/queue_trocr_staged.out 2>&1 &

set -u

cd /home/jupyter/OCC_HTR
export PATH=$HOME/.local/bin:$PATH
mkdir -p logs/finetune_ocr logs/trocr_finetune

# --- shared knobs ---
REAL_FOLDER="./data/processed/annotated_samples/OCR/full_annotated"
AUG_MED="./data/processed/synthetic_samples/augmented_images/aug_20260721_v2_medical"
LBL_MED="./data/processed/synthetic_samples/img_labels/labels_20260721_v2_medical/labels.json"
AUG_COM="./data/processed/synthetic_samples/augmented_images/aug_20260721_v2_matched_cometa"
LBL_COM="./data/processed/synthetic_samples/img_labels/labels_20260721_v2_matched_cometa/labels.json"
AUG_30K="./data/processed/synthetic_samples/augmented_images/aug_20260714_cometa_30k"
LBL_30K="./data/processed/synthetic_samples/img_labels/labels_20260714_cometa_30k/labels.json"

DRIVER_TS=$(date +%Y%m%d_%H%M%S)
DRIVER_LOG="logs/queue_trocr_staged_${DRIVER_TS}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER_LOG"; }

# run_phase: launch a python command line, redirecting stdout+stderr to its own log.
# We use `uv run` so the child inherits the uv-managed .venv (rapidfuzz, torch,
# transformers, etc). PROJECT_ROOT and PYTHONPATH are set so the src/ imports work.
run_phase() {
    local name="$1"; shift
    local log_path="$1"; shift
    log "=== START phase: $name  -> $log_path"
    env PROJECT_ROOT=. PYTHONPATH=. uv run python3 "$@" > "$log_path" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "=== END   phase: $name  OK (exit 0)"
    else
        log "=== END   phase: $name  FAILED (exit $rc). Continuing to next phase."
    fi
}

log "Driver started (TrOCR + staged reruns). Kraken 1 + 2 assumed already done."

# -----------------------------------------------------------------------------
# Phase 1: TrOCR ViT+RoBERTa + Dataset A''
# -----------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "trocr_vitroberta_matched_cometa" \
    "logs/trocr_finetune/trocr_vitroberta_matched_cometa_${TS}.out" \
    scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --pretrained-model-id microsoft/trocr-base-handwritten \
        --augmented-folder "$AUG_COM" \
        --labels-json "$LBL_COM" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# -----------------------------------------------------------------------------
# Phase 2: TrOCR ViT+RoBERTa + Dataset B''
# -----------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "trocr_vitroberta_medical" \
    "logs/trocr_finetune/trocr_vitroberta_medical_${TS}.out" \
    scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --pretrained-model-id microsoft/trocr-base-handwritten \
        --augmented-folder "$AUG_MED" \
        --labels-json "$LBL_MED" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# -----------------------------------------------------------------------------
# Phase 3: TrOCR Swin+BERT single-stage + Dataset A''
# -----------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "trocr_swinbert_singlestage_matched_cometa" \
    "logs/trocr_finetune/trocr_swinbert_singlestage_matched_cometa_${TS}.out" \
    scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --augmented-folder "$AUG_COM" \
        --labels-json "$LBL_COM" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# -----------------------------------------------------------------------------
# Phase 4: TrOCR Swin+BERT single-stage + Dataset B''
# -----------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "trocr_swinbert_singlestage_medical" \
    "logs/trocr_finetune/trocr_swinbert_singlestage_medical_${TS}.out" \
    scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --augmented-folder "$AUG_MED" \
        --labels-json "$LBL_MED" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# -----------------------------------------------------------------------------
# Phase 5: Stage 1a — Swin+BERT pretrain on 30k COMETA
# -----------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
STAGE1_LOG="logs/trocr_finetune/stage1a_swinbert_cometa_30k_pretrain_${TS}.out"
run_phase "stage1a_swinbert_cometa_30k_pretrain" \
    "$STAGE1_LOG" \
    scripts/ocr/run_trocr_finetune.py \
        --augmented-folder "$AUG_30K" \
        --labels-json "$LBL_30K" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.05 --seed 42 --epochs 15 \
        --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 4 --dataloader-num-workers 4 --device cuda

# Locate the Stage 1a best_model dir for downstream stages
STAGE1_RUN_DIR=$(ls -td models/ocr/finetuned/trocr_* 2>/dev/null | head -1)
STAGE1_BEST="${STAGE1_RUN_DIR}/best_model"
log "Stage 1a best_model = $STAGE1_BEST"

if [ ! -d "$STAGE1_BEST" ]; then
    log "ERROR: Stage 1a best_model not found; skipping Stage 2a/2b. Investigate $STAGE1_LOG."
    exit 1
fi

# -----------------------------------------------------------------------------
# Phase 6: Stage 2a — fine-tune on A''
# -----------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "stage2a_swinbert_matched_cometa" \
    "logs/trocr_finetune/stage2a_swinbert_matched_cometa_${TS}.out" \
    scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --pretrained-model-id "$STAGE1_BEST" \
        --augmented-folder "$AUG_COM" \
        --labels-json "$LBL_COM" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# -----------------------------------------------------------------------------
# Phase 7: Stage 2b — fine-tune on B''
# -----------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "stage2b_swinbert_medical" \
    "logs/trocr_finetune/stage2b_swinbert_medical_${TS}.out" \
    scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --pretrained-model-id "$STAGE1_BEST" \
        --augmented-folder "$AUG_MED" \
        --labels-json "$LBL_MED" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

log "=== ALL PHASES COMPLETE ==="
log "Driver log: $DRIVER_LOG"
