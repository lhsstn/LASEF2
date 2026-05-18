#!/bin/bash
PRJ_PATH=/home/seoultech/MLP/hslim/SCALE
source "$PRJ_PATH/.scale/bin/activate"

# ==============================
# GPUs
# ==============================
GPU_START=4
GPU_COUNT=4
CUDA_VISIBLE_DEVICES=$(seq -s, $GPU_START $((GPU_START + GPU_COUNT - 1)))
echo "🟢 Using GPUs: $CUDA_VISIBLE_DEVICES"
echo

# ==============================
# Datasets
# ==============================
MGSM=limhyeonseok/mgsm-all-low-resource-translated
# MGSM=limhyeonseok/mgsm-low-resource-translated
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
# BATCH=16
BATCH=32
SAMPLING=1


METHOD="cot"
TRANSLATE_Q="yes"       # ko/sw 질문을 영어로 번역
TRANSLATE_COT=""        # 영어 CoT 생성 후 target 언어로 번역 (필요시 "yes"로 변경)

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

    # Output filename includes method and translation options
    local SUFFIX=""
    [[ -n "$TRANSLATE_Q_FLAG" ]] && SUFFIX="${SUFFIX}_transQ"
    [[ -n "$TRANSLATE_COT_FLAG" ]] && SUFFIX="${SUFFIX}_transCOT"

    local ROLLOUT_FLAG=""
    local ROLLOUT_SUFFIX=""
    if [[ -n "$ROLLOUT" ]] && [[ "$ROLLOUT" -gt 1 ]]; then
        ROLLOUT_FLAG="--rollout $ROLLOUT"
        ROLLOUT_SUFFIX="_rollout${ROLLOUT}"
    fi

    local OUTPUT="$PRJ_PATH/data/inference_result/eval/${DATASET_NAME}/${MODEL_NAME}-${METHOD}${SUFFIX}-Google-transQ${ROLLOUT_SUFFIX}.jsonl"

    # Reasoning model flag
    local REASONING_FLAG=""
    if [[ "$MODEL_KIND" == "reasoning" ]]; then
        REASONING_FLAG="--reasoning_model"
    fi

    # Modes flag (only for CoT)
    local MODES_FLAG=""
    if [[ -n "$MODES" ]]; then
        MODES_FLAG="--modes $MODES"
    fi

    # Translation flags
    local TRANS_Q=""
    local TRANS_COT=""
    [[ -n "$TRANSLATE_Q_FLAG" ]] && TRANS_Q="--translate_q"
    [[ -n "$TRANSLATE_COT_FLAG" ]] && TRANS_COT="--translate_cot"

    echo "🚀 Running inference"
    echo "   ➜ Method       : $METHOD"
    echo "   ➜ Model        : $MODEL_NAME"
    echo "   ➜ Dataset      : $DATASET_NAME"
    echo "   ➜ Langs        : $LANGS"
    echo "   ➜ Translate Q  : ${TRANSLATE_Q_FLAG:-No}"
    echo "   ➜ Translate CoT: ${TRANSLATE_COT_FLAG:-No}"
    echo "   ➜ Output       : $OUTPUT"
    echo

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 "$PRJ_PATH/scripts/src/eval/eval_combined_TQ_low2.py" \
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
# 평가할 모델 목록
# ==============================
MODELS=(
    # "google/gemma-3-4b-it"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "google/gemma-3-27b-it"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # "allenai/OLMo-2-1124-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)


# # ==============================
# # Langs & Modes 설정
# # ==============================
ALL_LANGS="ta,kn,my,km,am,yo,si,gu,ne,uz,ky,ceb,eu,gn,hy,jv,ka,kk,ku,lo,mg,ml,mn,mr,mt,or,pa,ps,qu,sd,so,su,tg,ug"
ALL_MODES="ta-ta-nonskeleton,kn-kn-nonskeleton,my-my-nonskeleton,km-km-nonskeleton,am-am-nonskeleton,yo-yo-nonskeleton,si-si-nonskeleton,gu-gu-nonskeleton,ne-ne-nonskeleton,uz-uz-nonskeleton,ky-ky-nonskeleton,ceb-ceb-nonskeleton,eu-eu-nonskeleton,gn-gn-nonskeleton,hy-hy-nonskeleton,jv-jv-nonskeleton,ka-ka-nonskeleton,kk-kk-nonskeleton,ku-ku-nonskeleton,lo-lo-nonskeleton,mg-mg-nonskeleton,ml-ml-nonskeleton,mn-mn-nonskeleton,mr-mr-nonskeleton,mt-mt-nonskeleton,or-or-nonskeleton,pa-pa-nonskeleton,ps-ps-nonskeleton,qu-qu-nonskeleton,sd-sd-nonskeleton,so-so-nonskeleton,su-su-nonskeleton,tg-tg-nonskeleton,ug-ug-nonskeleton"

echo "📌 Models       : ${#MODELS[@]}"
echo "📌 Datasets     : ${#DATASETS[@]}"
echo "📌 Langs        : $ALL_LANGS"
echo

# ==============================
# 실행: 모든 모델 × 모든 데이터셋
# ==============================
for MODEL in "${MODELS[@]}"; do
    for DATA in "${DATASETS[@]}"; do
        run_inference "$MODEL" "cot" "$ALL_LANGS" "$DATA" "instruct" "$ALL_MODES"
    done
done

echo "🎉 ALL skeleton evaluations finished!"