#!/bin/bash
PRJ_PATH=/home/work/mlp/hslim/LASEF2
source "$PRJ_PATH/.lasef/bin/activate"

# ==============================
# GPUs
# ==============================
GPU_START=0
GPU_COUNT=4
CUDA_VISIBLE_DEVICES=$(seq -s, $GPU_START $((GPU_START + GPU_COUNT - 1)))
echo "🟢 Using GPUs: $CUDA_VISIBLE_DEVICES"
echo

# ==============================
# Inference defaults
# ==============================
TP=$GPU_COUNT
MAXTOK=4096
TEMP=0.0
TOPP=1
GPU_MEM=0.92
DTYPE=bfloat16
BATCH=16
ROLLOUT=10

run_inference() {
    local MODEL_PATH=$1
    local SKEL_LANG=$2

    local MODEL_NAME
    MODEL_NAME=$(basename "$MODEL_PATH")

    local ROLLOUT_FLAG=""
    local ROLLOUT_SUFFIX=""
    if [[ -n "$ROLLOUT" ]] && [[ "$ROLLOUT" -gt 1 ]]; then
        ROLLOUT_FLAG="--rollout $ROLLOUT"
        ROLLOUT_SUFFIX="_rollout${ROLLOUT}"
    fi

    # Determine translated skeleton input file
    local SKELETON_FILE
    if [[ "$SKEL_LANG" == "en" ]]; then
        SKELETON_FILE="$PRJ_PATH/data/results/SkelLang/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl"
    else
        SKELETON_FILE="$PRJ_PATH/data/results/SkelLang/translated_skeleton/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-${SKEL_LANG}_translated.jsonl"
    fi

    # Check if skeleton file exists
    if [[ ! -f "$SKELETON_FILE" ]]; then
        echo "⚠️ Skeleton file not found: $SKELETON_FILE. Skipping."
        return
    fi

    local OUTPUT="$PRJ_PATH/data/results/SkelLang/mgsm-all-low-resource-translated/${MODEL_NAME}-skeleton_multiturn_skelLang-${SKEL_LANG}_translated_solved${ROLLOUT_SUFFIX}.jsonl"

    echo "🚀 Running inference using translated skeleton"
    echo "   ➜ Model          : $MODEL_NAME"
    echo "   ➜ Skeleton Lang  : $SKEL_LANG"
    echo "   ➜ Skeleton Input : $SKELETON_FILE"
    echo "   ➜ Output         : $OUTPUT"
    echo

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PRJ_PATH/scripts/src/eval_with_translated_skeleton.py" \
        --skeleton_file "$SKELETON_FILE" \
        --model "$MODEL_PATH" \
        --output "$OUTPUT" \
        --tp "$TP" \
        --batch "$BATCH" \
        --max_tokens "$MAXTOK" \
        --temp "$TEMP" \
        --top_p "$TOPP" \
        --gpu_mem "$GPU_MEM" \
        --dtype "$DTYPE" \
        $ROLLOUT_FLAG
}

# ==============================
# Models
# ==============================
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

# ==============================
# Skeleton Languages to Test
# ==============================
SKELETON_LANGS=(
    "en"
    "zh"  # Chinese
    "ru"  # Russian
    "es"  # Spanish
    "ko"  # Korean
    "th"  # Thai
)

echo "📌 Models          : ${#MODELS[@]}"
echo "📌 Skeleton Langs  : ${SKELETON_LANGS[*]}"
echo

# ==============================
# Run: All Models × All Skeleton Languages
# ==============================
for SKEL_LANG in "${SKELETON_LANGS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📝 Skeleton Language: $SKEL_LANG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for MODEL in "${MODELS[@]}"; do
        run_inference "$MODEL" "$SKEL_LANG"
    done
done

echo "🎉 ALL translated skeleton evaluations finished!"
