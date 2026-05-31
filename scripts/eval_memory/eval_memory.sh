#!/bin/bash
# Evaluate memory-augmented Motus policy on RoboTwin (all 50 tasks)
# Uses deploy_policy_memory.py via environment variable toggle in deploy_policy.py.
#
# Usage:
#   bash scripts/eval_memory/eval_memory.sh <injector_checkpoint> [options]
#
# Options:
#   --tasks <file>    Task list file (default: tasks_all.txt)
#   --soft            Use RecencyWeightedRetriever instead of FIFO top-k
#   --bank_size <n>   Memory bank size (default: 5)
#   --top_k <n>       Top-k for retrieval (default: 5)
#   --gpus <ids>      Comma-separated GPU IDs (default: auto-detect)
#
# Examples:
#   bash scripts/eval_memory/eval_memory.sh results/injector_best.pt
#   bash scripts/eval_memory/eval_memory.sh results/injector_soft_best.pt --soft
#   bash scripts/eval_memory/eval_memory.sh results/injector_best.pt --tasks tasks_long_horizon.txt

set -e

# --- Parse arguments ---
INJECTOR_CKPT=""
TASKS_FILE=""
USE_SOFT=false
BANK_SIZE=5
TOP_K=5
GPU_IDS_STR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks) TASKS_FILE="$2"; shift 2 ;;
        --soft) USE_SOFT=true; shift ;;
        --bank_size) BANK_SIZE="$2"; shift 2 ;;
        --top_k) TOP_K="$2"; shift 2 ;;
        --gpus) GPU_IDS_STR="$2"; shift 2 ;;
        *) INJECTOR_CKPT="$1"; shift ;;
    esac
done

if [ -z "$INJECTOR_CKPT" ]; then
    echo "Usage: bash eval_memory.sh <injector_checkpoint> [--tasks file] [--soft] [--bank_size N] [--top_k N] [--gpus 0,1,2,3]"
    exit 1
fi

# Get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTUS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POLICY_DIR="${MOTUS_ROOT}/inference/robotwin/Motus"
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: paths_config.yml not found at $CONFIG_FILE"
    exit 1
fi

# Parse paths_config.yml
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
POLICY_NAME="Motus"

# Resolve injector checkpoint
if [ ! -f "$INJECTOR_CKPT" ]; then
    INJECTOR_CKPT="${MOTUS_ROOT}/${INJECTOR_CKPT}"
fi
if [ ! -f "$INJECTOR_CKPT" ]; then
    echo "Error: Injector checkpoint not found: $INJECTOR_CKPT"
    exit 1
fi
INJECTOR_CKPT=$(realpath "$INJECTOR_CKPT")

# Tasks file
if [ -z "$TASKS_FILE" ]; then
    TASKS_FILE="${POLICY_DIR}/tasks_all.txt"
elif [ ! -f "$TASKS_FILE" ]; then
    TASKS_FILE="${POLICY_DIR}/${TASKS_FILE}"
fi
if [ ! -f "$TASKS_FILE" ]; then
    echo "Error: Tasks file not found: $TASKS_FILE"
    exit 1
fi

# GPU setup
if [ -n "$GPU_IDS_STR" ]; then
    IFS=',' read -ra GPU_IDS <<< "$GPU_IDS_STR"
elif command -v nvidia-smi &> /dev/null; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    echo "Auto-detected ${#GPU_IDS[@]} GPUs: ${GPU_IDS[*]}"
else
    GPU_IDS=(0)
fi

# Environment setup
cd "$ROBOTWIN_ROOT" || exit 1
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8

# Export memory config as environment variables (read by deploy_policy.py)
export MOTUS_INJECTOR_CKPT="${INJECTOR_CKPT}"
export MOTUS_BANK_SIZE="${BANK_SIZE}"
export MOTUS_TOP_K="${TOP_K}"
if [ "$USE_SOFT" = true ]; then
    export MOTUS_USE_SOFT="1"
else
    export MOTUS_USE_SOFT="0"
fi

# Logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${POLICY_DIR}/logs_memory_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

mapfile -t tasks < "$TASKS_FILE"
total=${#tasks[@]}

echo ""
echo "================================================================"
echo " Memory-Augmented Motus Evaluation"
echo "================================================================"
echo " Injector:    $INJECTOR_CKPT"
echo " Bank size:   $BANK_SIZE"
echo " Top-k:       $TOP_K"
echo " Soft:        $USE_SOFT"
echo " Tasks:       $total"
echo " GPUs:        ${GPU_IDS[*]}"
echo " Log dir:     $LOG_DIR"
echo "================================================================"
echo ""

# GPU management
declare -A gpu_pid
for gpu_id in "${GPU_IDS[@]}"; do gpu_pid[$gpu_id]=""; done
is_running() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
get_free_gpu() {
    while true; do
        for gpu_id in "${GPU_IDS[@]}"; do
            if ! is_running "${gpu_pid[$gpu_id]}"; then echo "$gpu_id"; return 0; fi
        done
        sleep 2
    done
}
show_progress() {
    local current=$1 total=$2
    local percent=$((current * 100 / total))
    local bar_length=50 filled=$((percent * bar_length / 100))
    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%$((bar_length - filled))s" | tr ' ' ' '
    printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
}

# Launch tasks
pids=()
completed=0
echo -e "\033[32mLaunching evaluation tasks...\033[0m"

for task in "${tasks[@]}"; do
    gpu_id=$(get_free_gpu)
    log_file="${LOG_DIR}/${task}.log"
    echo -e "\033[36m→ Task: $task | GPU: $gpu_id\033[0m"

    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        PYTHONWARNINGS=ignore::UserWarning \
        python script/eval_policy.py \
            --config "policy/${POLICY_NAME}/deploy_policy.yml" \
            --overrides \
            --task_name "${task}" \
            --task_config "${TASK_CONFIG}" \
            --ckpt_setting "${CHECKPOINT_PATH}" \
            --seed "${SEED}" \
            --policy_name "${POLICY_NAME}" \
            --log_dir "${LOG_DIR}" \
            --wan_path "${WAN_PATH}" \
            --vlm_path "${VLM_PATH}" \
            > "$log_file" 2>&1
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "✓ Task $task completed successfully" >> "$log_file"
        else
            echo "✗ Task $task failed with exit code $exit_code" >> "$log_file"
        fi
    ) &

    pid=$!
    gpu_pid[$gpu_id]=$pid
    pids+=($pid)
    sleep 1
done

echo -e "\n\033[33mWaiting for completion...\033[0m"
for pid in "${pids[@]}"; do
    wait "$pid"
    ((completed++))
    show_progress $completed $total
done
echo -e "\n\033[32m✓ All tasks completed!\033[0m"

# Generate summary
summary="${LOG_DIR}/evaluation_summary.txt"
cat > "$summary" << EOF
Motus Memory-Augmented Evaluation Summary
==========================================
Date: $(date)
Host: $(hostname)
Injector: $INJECTOR_CKPT
Bank Size: $BANK_SIZE / Top-k: $TOP_K / Soft: $USE_SOFT
RoboTwin: $ROBOTWIN_ROOT
Checkpoint: $CHECKPOINT_PATH
Total Tasks: $total / GPUs: ${GPU_IDS[*]}

Task Results:
-------------
EOF

success=0; failed=0
for task in "${tasks[@]}"; do
    log_file="${LOG_DIR}/${task}.log"
    if [ ! -f "$log_file" ]; then
        echo "  ⚠️  $task: LOG NOT FOUND" >> "$summary"
        ((failed++))
    elif grep -q "Success rate:" "$log_file" 2>/dev/null; then
        last_line=$(grep -i "Success rate:" "$log_file" | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
        score=$(echo "$last_line" | grep -oP '=>\s*\K\d+\.?\d*(?=%)' | tail -1)
        if [ -n "$score" ]; then
            echo "  ✅ $task: ${score}%" >> "$summary"
        else
            echo "  ✅ $task: SUCCESS" >> "$summary"
        fi
        ((success++))
    elif grep -q "completed successfully\|Episode.*completed" "$log_file" 2>/dev/null; then
        echo "  ✅ $task: SUCCESS" >> "$summary"
        ((success++))
    else
        echo "  ❌ $task: FAILED" >> "$summary"
        ((failed++))
    fi
done

cat >> "$summary" << EOF

Summary:
--------
✅ Successful: $success / ❌ Failed: $failed / Total: $total
Success Rate: $(awk "BEGIN {printf \"%.1f\", $success * 100.0 / $total}")%
Logs: $LOG_DIR
EOF

echo ""
echo "================================================================"
cat "$summary"
echo "================================================================"

# Save failed tasks
failed_tasks_file="${LOG_DIR}/failed_tasks.txt"
> "$failed_tasks_file"
for task in "${tasks[@]}"; do
    log_file="${LOG_DIR}/${task}.log"
    if [ ! -f "$log_file" ] || ! grep -q "Success rate:\|completed successfully" "$log_file" 2>/dev/null; then
        echo "$task" >> "$failed_tasks_file"
    fi
done

exit 0
