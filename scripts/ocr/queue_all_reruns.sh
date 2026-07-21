#!/bin/bash
# Sequential driver: queues 8 training runs after kraken Run 1 completes.
#
# Runs in this order (single L4 GPU, so strictly sequential):
#   2. kraken matched-pool + medical
#   3. TrOCR ViT+RoBERTa + Dataset A'' (matched_cometa)
#   4. TrOCR ViT+RoBERTa + Dataset B'' (medical)
#   5. TrOCR Swin+BERT single-stage + Dataset A''
#   6. TrOCR Swin+BERT single-stage + Dataset B''
#   7. Swin+BERT Stage 1a — 30k COMETA pretrain
#   8. Swin+BERT Stage 2a — fine-tune Stage 1a on A''
#   9. Swin+BERT Stage 2b — fine-tune Stage 1a on B''
#
# Each run's stdout+stderr lands in its own file under logs/{finetune_ocr,trocr_finetune}/
# A phase marker log at logs/queue_all_reruns_<TS>.log records start/end + exit code of each phase.
#
# Usage on the VM (inside tmux):
#   cd /home/jupyter/OCC_HTR
#   nohup bash scripts/ocr/queue_all_reruns.sh > logs/queue_driver.out 2>&1 &
#   # ctrl-b d to detach, come back tomorrow.
set -u

cd /home/jupyter/OCC_HTR
export PATH=$HOME/.local/bin:$PATH
mkdir -p logs/finetune_ocr logs/trocr_finetune

# --- shared knobs ---
REAL_FOLDER="./data/processed/annotated_samples/OCR/full_annotated"
AUG_BASE="./data/processed/synthetic_samples/augmented_images/aug_20260721_121550"
LBL_BASE="./data/processed/synthetic_samples/img_labels/labels_20260721_121550/labels.json"
AUG_MED="./data/processed/synthetic_samples/augmented_images/aug_20260721_v2_medical"
LBL_MED="./data/processed/synthetic_samples/img_labels/labels_20260721_v2_medical/labels.json"
AUG_COM="./data/processed/synthetic_samples/augmented_images/aug_20260721_v2_matched_cometa"
LBL_COM="./data/processed/synthetic_samples/img_labels/labels_20260721_v2_matched_cometa/labels.json"
AUG_30K="./data/processed/synthetic_samples/augmented_images/aug_20260714_cometa_30k"
LBL_30K="./data/processed/synthetic_samples/img_labels/labels_20260714_cometa_30k/labels.json"

DRIVER_TS=$(date +%Y%m%d_%H%M%S)
DRIVER_LOG="logs/queue_all_reruns_${DRIVER_TS}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER_LOG"; }

run_phase() {
    local name="$1"; shift
    local log_path="$1"; shift
    log "=== START phase: $name  -> $log_path"
    "$@" > "$log_path" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "=== END   phase: $name  OK (exit 0)"
    else
        log "=== END   phase: $name  FAILED (exit $rc). Continuing to next phase."
    fi
}

# ---------------------------------------------------------------------------
# Phase 0: wait for kraken Run 1 (already launched interactively) to finish.
# We detect it by looking for a running finetune_ocr process on aug_20260721_121550
# ---------------------------------------------------------------------------
log "Driver started. Waiting for kraken Run 1 (no-medical) to finish before queuing rest..."
while pgrep -af "run_finetune_ocr.py.*aug_20260721_121550" > /dev/null; do
    sleep 60
done
log "Kraken Run 1 no longer running. Proceeding with queued runs."

# ---------------------------------------------------------------------------
# Phase 1: Kraken matched-pool + medical
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "kraken_600real_3000reren_1000medical" \
    "logs/finetune_ocr/kraken_600real_3000reren_1000medical_${TS}.out" \
    env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/ocr/run_finetune_ocr.py \
        --augmented-folder "$AUG_MED" \
        --labels-json "$LBL_MED" \
        --real-folder "$REAL_FOLDER" \
        --real-train-frac 0.8 --real-val-frac 0.2 \
        --seed 42 \
        --lrate 1e-5 --lag 5 --epochs -1 \
        --resize union \
        --device cuda:0

# ---------------------------------------------------------------------------
# Phase 2: TrOCR ViT+RoBERTa + Dataset A'' (matched_cometa)
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "trocr_vitroberta_matched_cometa" \
    "logs/trocr_finetune/trocr_vitroberta_matched_cometa_${TS}.out" \
    env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --pretrained-model-id microsoft/trocr-base-handwritten \
        --augmented-folder "$AUG_COM" \
        --labels-json "$LBL_COM" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# ---------------------------------------------------------------------------
# Phase 3: TrOCR ViT+RoBERTa + Dataset B'' (medical)
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "trocr_vitroberta_medical" \
    "logs/trocr_finetune/trocr_vitroberta_medical_${TS}.out" \
    env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --pretrained-model-id microsoft/trocr-base-handwritten \
        --augmented-folder "$AUG_MED" \
        --labels-json "$LBL_MED" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# ---------------------------------------------------------------------------
# Phase 4: TrOCR Swin+BERT single-stage + Dataset A''
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "trocr_swinbert_singlestage_matched_cometa" \
    "logs/trocr_finetune/trocr_swinbert_singlestage_matched_cometa_${TS}.out" \
    env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --augmented-folder "$AUG_COM" \
        --labels-json "$LBL_COM" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# ---------------------------------------------------------------------------
# Phase 5: TrOCR Swin+BERT single-stage + Dataset B'' (medical)
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "trocr_swinbert_singlestage_medical" \
    "logs/trocr_finetune/trocr_swinbert_singlestage_medical_${TS}.out" \
    env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --augmented-folder "$AUG_MED" \
        --labels-json "$LBL_MED" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# ---------------------------------------------------------------------------
# Phase 6: Stage 1a — Swin+BERT pretrain on 30k COMETA
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
STAGE1_LOG="logs/trocr_finetune/stage1a_swinbert_cometa_30k_pretrain_${TS}.out"
run_phase "stage1a_swinbert_cometa_30k_pretrain" \
    "$STAGE1_LOG" \
    env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/ocr/run_trocr_finetune.py \
        --augmented-folder "$AUG_30K" \
        --labels-json "$LBL_30K" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.05 --seed 42 --epochs 15 \
        --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 4 --dataloader-num-workers 4 --device cuda

# Locate the Stage 1a best_model dir for downstream stages
STAGE1_RUN_DIR=$(ls -td models/ocr/finetuned/trocr_* | head -1)
STAGE1_BEST="${STAGE1_RUN_DIR}/best_model"
log "Stage 1a best_model = $STAGE1_BEST"

if [ ! -d "$STAGE1_BEST" ]; then
    log "ERROR: Stage 1a best_model not found; skipping Stage 2a/2b. Investigate $STAGE1_LOG."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 7: Stage 2a — fine-tune Stage 1a on Dataset A'' (matched_cometa)
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "stage2a_swinbert_matched_cometa" \
    "logs/trocr_finetune/stage2a_swinbert_matched_cometa_${TS}.out" \
    env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/ocr/run_trocr_finetune.py \
        --real-folder "$REAL_FOLDER" \
        --pretrained-model-id "$STAGE1_BEST" \
        --augmented-folder "$AUG_COM" \
        --labels-json "$LBL_COM" \
        --output-base-dir ./models/ocr/finetuned \
        --val-fraction 0.2 --seed 42 --epochs 20 --learning-rate 5e-5 \
        --batch-size 32 --eval-batch-size 32 --max-target-length 128 --num-beams 4 \
        --early-stopping-patience 5 --dataloader-num-workers 4 --device cuda

# ---------------------------------------------------------------------------
# Phase 8: Stage 2b — fine-tune Stage 1a on Dataset B'' (medical)
# ---------------------------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
run_phase "stage2b_swinbert_medical" \
    "logs/trocr_finetune/stage2b_swinbert_medical_${TS}.out" \
    env PROJECT_ROOT=. PYTHONPATH=. python3 scripts/ocr/run_trocr_finetune.py \
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
