#!/bin/bash
PRJ_PATH=/home/work/mlp/hslim/LASEF2
source "$PRJ_PATH/.lasef/bin/activate"

# ==============================
# GPUs
# ==============================
GPU_START=0
GPU_COUNT=4
CUDA_VISIBLE_DEVICES=$(seq -s, $GPU_START $((GPU_START + GPU_COUNT - 1)))
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo

# ==============================
# Datasets
# ==============================
MGSM=limhyeonseok/MGSM
AIME=limhyeonseok/AIME25-translated
MATH=limhyeonseok/MATH-500-translated
POLYMATH=limhyeonseok/PolyMath-translated

DATASETS=(
    "$MGSM"
    "$MATH"
    "$POLYMATH"
)

# ==============================
# Inference defaults
# ==============================
TP=$GPU_COUNT
MAXTOK=8192
TEMP=0.0
TOPP=1
GPU_MEM=0.92
DTYPE=bfloat16
BATCH=32
SAMPLING=1
ROLLOUT=1


METHOD="cot"
TRANSLATE_Q=""
TRANSLATE_COT=""

run_inference() {
    local MODEL_PATH=$1
    local METHOD=$2
    local LANGS=$3
    local DATASET=$4
    local MODEL_KIND=${5:-instruct}
    local MODES=${6:-""}
    local TRANSLATE_Q_FLAG=${7:-""}
    local TRANSLATE_COT_FLAG=${8:-""}

    local MODEL_NAME
    MODEL_NAME=$(basename "$MODEL_PATH")

    local DATASET_NAME
    DATASET_NAME=$(basename "$DATASET")

    local SUFFIX=""
    [[ -n "$TRANSLATE_Q_FLAG" ]] && SUFFIX="${SUFFIX}_transQ"
    [[ -n "$TRANSLATE_COT_FLAG" ]] && SUFFIX="${SUFFIX}_transCOT"

    local ROLLOUT_FLAG=""
    local ROLLOUT_SUFFIX=""
    if [[ -n "$ROLLOUT" ]] && [[ "$ROLLOUT" -gt 1 ]]; then
        ROLLOUT_FLAG="--rollout $ROLLOUT"
        ROLLOUT_SUFFIX="_rollout${ROLLOUT}"
    fi

    local OUTPUT="$PRJ_PATH/data/results/llama/${DATASET_NAME}/${MODEL_NAME}-${METHOD}${SUFFIX}${ROLLOUT_SUFFIX}.jsonl"

    local REASONING_FLAG=""
    if [[ "$MODEL_KIND" == "reasoning" ]]; then
        REASONING_FLAG="--reasoning_model"
    fi

    local MODES_FLAG=""
    if [[ -n "$MODES" ]]; then
        MODES_FLAG="--modes $MODES"
    fi

    local TRANS_Q=""
    local TRANS_COT=""
    [[ -n "$TRANSLATE_Q_FLAG" ]] && TRANS_Q="--translate_q"
    [[ -n "$TRANSLATE_COT_FLAG" ]] && TRANS_COT="--translate_cot"

    echo "Running inference"
    echo "   Method       : $METHOD"
    echo "   Model        : $MODEL_NAME"
    echo "   Dataset      : $DATASET_NAME"
    echo "   Langs        : $LANGS"
    echo "   Translate Q  : ${TRANSLATE_Q_FLAG:-No}"
    echo "   Translate CoT: ${TRANSLATE_COT_FLAG:-No}"
    echo "   Output       : $OUTPUT"
    echo

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PRJ_PATH/scripts/src/eval_combined_nonTQ_llama.py" \
        --method "$METHOD" \
        --model "$MODEL_PATH" \
        --dataset "$DATASET" \
        --output "$OUTPUT" \
        --tp "$TP" \
        --batch "$BATCH" \
        --max_tokens "$MAXTOK" \
        --temp "$TEMP" \
        --top_p "$TOPP" \
        --gpu_mem "$GPU_MEM" \
        --dtype "$DTYPE" \
        --langs "$LANGS" \
        --sample_ratio $SAMPLING \
        $ROLLOUT_FLAG \
        $MODES_FLAG \
        $TRANS_Q \
        $TRANS_COT \
        $REASONING_FLAG
}

# ==============================
# Llama 3.1 models
# ==============================
MODELS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "meta-llama/Meta-Llama-3.1-70B-Instruct"
)


# ==============================
# Langs & Modes
# ==============================
ALL_LANGS="en,zh,es,ko,th,sw,te,bn"
ALL_MODES="zh-zh-skeleton,es-es-skeleton,ko-ko-skeleton,th-th-skeleton,sw-sw-skeleton,te-te-skeleton"

echo "Models       : ${#MODELS[@]}"
echo "Datasets     : ${#DATASETS[@]}"
echo "Langs        : $ALL_LANGS"
echo

# # ==============================
# # CoT
# # ==============================
# for MODEL in "${MODELS[@]}"; do
#     for DATA in "${DATASETS[@]}"; do
#         run_inference "$MODEL" "cot" "$ALL_LANGS" "$DATA" "instruct" "$ALL_MODES"
#     done
# done

# echo "ALL nonTQ evaluations finished!"


# ALL_MODES="zh-zh-skeleton,es-es-skeleton,ko-ko-skeleton,th-th-skeleton,sw-sw-skeleton,te-te-skeleton,bn-bn-skeleton"

# echo "Models       : ${#MODELS[@]}"
# echo "Datasets     : ${#DATASETS[@]}"
# echo "Langs        : $ALL_LANGS"
# echo

# ==============================
# Skeleton Multi-turn
# ==============================
for MODEL in "${MODELS[@]}"; do
    for DATA in "${DATASETS[@]}"; do
        run_inference "$MODEL" "skeleton_multiturn" "$ALL_LANGS" "$DATA" "instruct" "$ALL_MODES"
    done
done

echo "ALL skeleton evaluations finished!"
