#!/bin/bash
# Train ActionMemoryInjector (Experiment 3.2)
# Freezes backbone, trains cross-attention injector on action tokens.
#
# Usage:
#   bash scripts/eval_memory/run_train_injector.sh              # multi-GPU training only
#   bash scripts/eval_memory/run_train_injector.sh --dry        # single-GPU smoke test
#   bash scripts/eval_memory/run_train_injector.sh --eval       # train + RoboTwin eval
#   bash scripts/eval_memory/run_train_injector.sh --eval_only  # skip training, eval only

set -e

cd /kpfs-intern/jialongliu/projects/Motus
PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/motus/bin/python
ACCELERATE=/kpfs-intern/jialongliu/miniforge3/envs/motus/bin/accelerate

# --- Configuration ---
CONFIG="configs/robotwin.yaml"
CHECKPOINT="pretrained_models/Motus_robotwin2/mp_rank_00_model_states.pt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="scripts/eval_memory/results"

# Training hyperparams
EPOCHS=10
LR=1e-4
NUM_TRAIN_BATCHES=200
NUM_EVAL_BATCHES=50
NUM_INFERENCE_STEPS=10
BANK_SIZE=5
TOP_K=5
EVAL_EVERY=50

# GPU config
NUM_GPUS=4
ACCELERATE_CONFIG="configs/accelerate/default_config.yaml"

# --- Parse args ---
DRY_RUN=false
RUN_EVAL=false
EVAL_ONLY=false
EVAL_CKPT=""

for arg in "$@"; do
    case $arg in
        --dry) DRY_RUN=true ;;
        --eval) RUN_EVAL=true ;;
        --eval_only) EVAL_ONLY=true ;;
        --eval_ckpt=*) EVAL_CKPT="${arg#*=}" ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN: single-GPU smoke test ==="
    NUM_GPUS=1
    EPOCHS=1
    NUM_TRAIN_BATCHES=3
    NUM_EVAL_BATCHES=2
    NUM_INFERENCE_STEPS=2
    EVAL_EVERY=2
    OUTPUT_DIR="${OUTPUT_BASE}/test_injector_${TIMESTAMP}"
else
    echo "=== Full training: ${NUM_GPUS} GPUs ==="
    OUTPUT_DIR="${OUTPUT_BASE}/injector_${TIMESTAMP}"
fi

# --- Training ---
if [ "$EVAL_ONLY" = false ]; then
    echo "Config: ${CONFIG}"
    echo "Checkpoint: ${CHECKPOINT}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Epochs: ${EPOCHS}, LR: ${LR}, Bank: ${BANK_SIZE}, Top-k: ${TOP_K}"
    echo "Train batches: ${NUM_TRAIN_BATCHES}, Eval batches: ${NUM_EVAL_BATCHES}"
    echo "Inference steps: ${NUM_INFERENCE_STEPS}"
    echo ""

    if [ "$NUM_GPUS" -gt 1 ]; then
        ${ACCELERATE} launch \
            --config_file ${ACCELERATE_CONFIG} \
            --num_processes ${NUM_GPUS} \
            scripts/eval_memory/train_injector.py \
            --config ${CONFIG} \
            --checkpoint ${CHECKPOINT} \
            --output_dir ${OUTPUT_DIR} \
            --num_epochs ${EPOCHS} \
            --lr ${LR} \
            --num_train_batches ${NUM_TRAIN_BATCHES} \
            --num_eval_batches ${NUM_EVAL_BATCHES} \
            --num_inference_steps ${NUM_INFERENCE_STEPS} \
            --bank_size ${BANK_SIZE} \
            --top_k ${TOP_K} \
            --eval_every_steps ${EVAL_EVERY}
    else
        ${PYTHON} scripts/eval_memory/train_injector.py \
            --config ${CONFIG} \
            --checkpoint ${CHECKPOINT} \
            --output_dir ${OUTPUT_DIR} \
            --num_epochs ${EPOCHS} \
            --lr ${LR} \
            --num_train_batches ${NUM_TRAIN_BATCHES} \
            --num_eval_batches ${NUM_EVAL_BATCHES} \
            --num_inference_steps ${NUM_INFERENCE_STEPS} \
            --bank_size ${BANK_SIZE} \
            --top_k ${TOP_K} \
            --eval_every_steps ${EVAL_EVERY}
    fi

    echo ""
    echo "=== Training Complete ==="
    echo "Output: ${OUTPUT_DIR}/"
    echo "Best checkpoint: ${OUTPUT_DIR}/injector_best.pt"
fi

# --- RoboTwin Evaluation ---
if [ "$RUN_EVAL" = true ] || [ "$EVAL_ONLY" = true ]; then
    # Determine checkpoint to evaluate
    if [ -n "$EVAL_CKPT" ]; then
        CKPT_TO_EVAL="$EVAL_CKPT"
    elif [ -f "${OUTPUT_DIR}/injector_best.pt" ]; then
        CKPT_TO_EVAL="${OUTPUT_DIR}/injector_best.pt"
    else
        echo "Error: No checkpoint found for evaluation. Use --eval_ckpt=<path>"
        exit 1
    fi

    echo ""
    echo "=== Starting RoboTwin Evaluation ==="
    echo "Checkpoint: ${CKPT_TO_EVAL}"

    if [ "$DRY_RUN" = true ]; then
        # Single task for smoke test
        bash scripts/eval_memory/eval_memory.sh "${CKPT_TO_EVAL}" \
            --tasks click_alarmclock --bank_size ${BANK_SIZE} --top_k ${TOP_K}
    else
        # Full 50-task evaluation
        bash scripts/eval_memory/eval_memory.sh "${CKPT_TO_EVAL}" \
            --bank_size ${BANK_SIZE} --top_k ${TOP_K}
    fi
fi
