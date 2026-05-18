#!/bin/bash
PRJ_PATH=/home/seoultech/MLP/hslim/SCALE
source "$PRJ_PATH/.scale/bin/activate"

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
TEMP=0.0
TOPP=1
GPU_MEM=0.92
DTYPE=bfloat16
BATCH=32
SAMPLING=1

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


    local OUTPUT="$PRJ_PATH/data/inference_result/eval_low_anal/${DATASET_NAME}/${MODEL_NAME}-${METHOD}_skelLang-${SKEL_LANG}${ROLLOUT_SUFFIX}.jsonl"

    echo "🚀 Running inference"
    echo "   ➜ Method         : $METHOD"
    echo "   ➜ Model          : $MODEL_NAME"
    echo "   ➜ Dataset        : $DATASET_NAME"
    echo "   ➜ Langs          : $LANGS"
    echo "   ➜ Skeleton Lang  : $SKEL_LANG"
    echo "   ➜ Output         : $OUTPUT"
    echo

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PRJ_PATH/scripts/src/eval/eval_combined_nonTQ_low_anal2.py" \
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
        --skeleton_lang "$SKEL_LANG" \
        --sample_ratio $SAMPLING
}

# ==============================
# Models
# ==============================
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # "meta-llama/Meta-Llama-3.1-70B-Instruct"
)

# ==============================
# Low-resource Languages to evaluate
# ==============================
# ALL_LANGS="kn,uz,ky"
# ALL_LANGS="ta,kn,my,km,am,yo,si,gu,ne,uz,ky"
ALL_LANGS="ta,kn,my,km,am,yo,si,gu,ne,uz,ky,ceb,eu,gn,hy,jv,ka,kk,ku,lo,mg,ml,mn,mr,mt,or,pa,ps,qu,sd,so,su,tg,ug"


# ==============================
# Skeleton Languages to Test
# Available: en, es, ko, zh, ru, th
# ==============================
SKELETON_LANGS=(
    # "zh"  # Chinese
    # "ru"  # Russian
    # "es"  # Spanish
    "ko"  # Korean
    "th"  # Thai
)

echo "📌 Models          : ${#MODELS[@]}"
echo "📌 Datasets        : ${#DATASETS[@]}"
echo "📌 Question Langs  : $ALL_LANGS"
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
            run_inference "$MODEL" "$METHOD" "$ALL_LANGS" "$DATA" "$SKEL_LANG"
        done
    done
done

echo "🎉 ALL low-resource skeleton evaluations finished!"
