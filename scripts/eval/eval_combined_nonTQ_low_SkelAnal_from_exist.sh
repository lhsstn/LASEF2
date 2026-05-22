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
# Datasets
# ==============================
# MGSM=limhyeonseok/mgsm-low-resource-translated
MGSM=limhyeonseok/mgsm-all-low-resource-translated

AIME=limhyeonseok/AIME25-translated
MATH=limhyeonseok/MATH-500-translated
POLYMATH=limhyeonseok/PolyMath-translated

DATASETS=(
    # "$MATH"
    # "$POLYMATH"
    "$MGSM"
)


# ==============================
# Inference defaults
# ==============================
TP=$GPU_COUNT
MAXTOK=4096
TEMP=0.7
TOPP=0.8
GPU_MEM=0.92
DTYPE=bfloat16
BATCH=32
ROLLOUT=10

METHOD="skeleton_multiturn"

run_inference() {
    local MODEL_PATH=$1
    local METHOD=$2
    local LANGS=$3
    local DATASET=$4
    local SKEL_LANG=${5:-"en"}

    local MODEL_NAME
    MODEL_NAME=$(basename "$MODEL_PATH")

    local DATASET_NAME
    DATASET_NAME=$(basename "$DATASET")

    local ROLLOUT_FLAG=""
    local ROLLOUT_SUFFIX=""
    if [[ -n "$ROLLOUT" ]] && [[ "$ROLLOUT" -gt 1 ]]; then
        ROLLOUT_FLAG="--rollout $ROLLOUT"
        ROLLOUT_SUFFIX="_rollout${ROLLOUT}"
    fi

    # Input skeleton file: from single_rollout directory
    local INPUT_SKEL="$PRJ_PATH/data/results/SkelLang/single_rollout/${MODEL_NAME}-${METHOD}_skelLang-${SKEL_LANG}.jsonl"
    local OUTPUT="$PRJ_PATH/data/results/SkelLang/${DATASET_NAME}/${MODEL_NAME}-${METHOD}_skelLang-${SKEL_LANG}${ROLLOUT_SUFFIX}_exist.jsonl"

    echo "🚀 Running inference from existing skeletons"
    echo "   ➜ Method         : $METHOD"
    echo "   ➜ Model          : $MODEL_NAME"
    echo "   ➜ Dataset        : $DATASET_NAME"
    echo "   ➜ Skeleton Lang  : $SKEL_LANG"
    echo "   ➜ Input Skel File: $INPUT_SKEL"
    echo "   ➜ Rollout        : $ROLLOUT"
    echo "   ➜ Output         : $OUTPUT"
    echo

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PRJ_PATH/scripts/src/eval_combined_nonTQ_low_SkelAnal_from_exist.py" \
        --model "$MODEL_PATH" \
        --skeleton_file "$INPUT_SKEL" \
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
# Available: en, es, ko, zh, ru, th
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
echo "📌 Datasets        : ${#DATASETS[@]}"
echo "📌 Skeleton Langs  : ${SKELETON_LANGS[*]}"
echo

# ==============================
# Run: All Models × All Datasets × All Skeleton Languages
# ==============================
for SKEL_LANG in "${SKELETON_LANGS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📝 Skeleton Language: $SKEL_LANG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for MODEL in "${MODELS[@]}"; do
        for DATA in "${DATASETS[@]}"; do
            run_inference "$MODEL" "$METHOD" "" "$DATA" "$SKEL_LANG"
        done
    done
done

echo "🎉 ALL low-resource skeleton evaluations finished!"
