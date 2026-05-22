#!/bin/bash
# ==============================================================================
# Batch Evaluation Runner for SkelLang Multi-Rollouts (MGSM)
# ==============================================================================

PRJ_PATH="/home/work/mlp/hslim/LASEF2"
PYTHON_BIN="${PRJ_PATH}/.fasttext/bin/python3"
SCRIPT_PATH="${PRJ_PATH}/compare_SkelLang_run.py"
MODEL="Qwen2.5-7B-Instruct"

# List of target datasets to evaluate
DATASETS=(
  "${PRJ_PATH}/data/results/SkelLang_MGSM/3_rollout"
  "${PRJ_PATH}/data/results/SkelLang_MGSM/5_rollout"
  "${PRJ_PATH}/data/results/SkelLang_MGSM/10_rollout"
  "${PRJ_PATH}/data/results/SkelLang_MGSM/10_rollout_exist"
  "${PRJ_PATH}/data/results/SkelLang_MGSM/single_rollout"
#   "${PRJ_PATH}/data/results/SkelLang_MGSM/translated_skeleton"
  "${PRJ_PATH}/data/results/SkelLang_MGSM/translated_skeleton_solved"
)

echo "========================================================"
echo " 🚀 Starting Batch Evaluation for Model: $MODEL"
echo "========================================================"

for DATA_DIR in "${DATASETS[@]}"; do
    if [ ! -d "$DATA_DIR" ]; then
        echo "⚠️  Dataset folder not found, skipping: $DATA_DIR"
        continue
    fi

    FOLDER_NAME=$(basename "$DATA_DIR")
    echo ""
    echo "========================================================"
    echo "▶ Evaluating: $FOLDER_NAME"
    echo "📂 Path: $DATA_DIR"
    echo "========================================================"

    # Run execution script with arguments
    "$PYTHON_BIN" "$SCRIPT_PATH" \
        --data_dir "$DATA_DIR" \
        --model "$MODEL" \
        --translated False \
        --rollout "auto" \
        --rollout_count "auto"

    echo "✔ Completed evaluation for $FOLDER_NAME"
done

echo ""
echo "========================================================"
echo "🎉 All Batch Evaluations Completed Successfully!"
echo "========================================================"
