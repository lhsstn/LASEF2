# LASEF2

python scripts/src/translate_skeleton_batch.py \
    --check \
    --batch_dir data/batch_requests/skeleton_translation


python scripts/src/translate_skeleton_batch.py \
    --download \
    --batch_dir data/batch_requests/skeleton_translation \
    --download_dir data/batch_requests/skeleton_translation/results


# 1. 중국어(zh) 후처리
python scripts/src/postprocess_translated_skeleton.py \
    --original data/results/SkelLang/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl \
    --batch_result data/batch_requests/skeleton_translation/results/batch_result_zh.jsonl \
    --target_lang zh \
    --output data/results/SkelLang/translated_skeleton/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-zh_translated.jsonl

# 2. 스페인어(es) 후처리
python scripts/src/postprocess_translated_skeleton.py \
    --original data/results/SkelLang/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl \
    --batch_result data/batch_requests/skeleton_translation/results/batch_result_es.jsonl \
    --target_lang es \
    --output data/results/SkelLang/translated_skeleton/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-es_translated.jsonl

# 3. 한국어(ko) 후처리
python scripts/src/postprocess_translated_skeleton.py \
    --original data/results/SkelLang/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl \
    --batch_result data/batch_requests/skeleton_translation/results/batch_result_ko.jsonl \
    --target_lang ko \
    --output data/results/SkelLang/translated_skeleton/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-ko_translated.jsonl

# 4. 태국어(th) 후처리
python scripts/src/postprocess_translated_skeleton.py \
    --original data/results/SkelLang/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl \
    --batch_result data/batch_requests/skeleton_translation/results/batch_result_th.jsonl \
    --target_lang th \
    --output data/results/SkelLang/translated_skeleton/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-th_translated.jsonl

# 5. 러시아어(ru) 후처리
python scripts/src/postprocess_translated_skeleton.py \
    --original data/results/SkelLang/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl \
    --batch_result data/batch_requests/skeleton_translation/results/batch_result_ru.jsonl \
    --target_lang ru \
    --output data/results/SkelLang/translated_skeleton/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-ru_translated.jsonl


bash scripts/eval/eval_with_translated_skeleton.sh
