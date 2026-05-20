Viewed postprocess_translated_skeleton.py:1-59

번역 파이프라인을 처음부터 끝까지 실행하는 전체 과정을 순서대로 정리해 드립니다. 터미널에서 차례대로 실행해 주시면 됩니다.

---

### 1단계: 가상환경 활성화 및 API Key 설정
먼저 가상환경을 켜고, OpenAI API Key를 환경변수로 등록합니다.

```bash
# 1. 프로젝트 폴더로 이동 (이미 이동해 계시다면 생략 가능)
cd /home/work/mlp/hslim/LASEF2

# 2. 가상환경 활성화
source .lasef/bin/activate

# 3. OpenAI API Key 설정 (본인의 실제 키를 입력해주세요)
export OPENAI_API_KEY="sk-proj-..."
```

---

### 2단계: GPT Batch 번역 요청 파일 생성 및 제출
영문 skeleton 파일을 읽어 5개 언어(`zh, es, ko, th, ru`)용 OpenAI Batch 요청용 `.jsonl` 파일들을 만들고 제출합니다.

```bash
# 1. 각 언어별 번역 Batch 요청 파일(.jsonl) 생성
python scripts/src/translate_skeleton_batch.py \
    --create \
    --input data/results/SkelLang/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl \
    --batch_dir data/batch_requests/skeleton_translation \
    --target_langs zh,es,ko,th,ru

# 2. 생성된 번역 Batch를 OpenAI 서버에 업로드 및 제출
python scripts/src/translate_skeleton_batch.py \
    --submit \
    --batch_dir data/batch_requests/skeleton_translation
```
*제출이 완료되면 각 언어별로 `batch_xxxx` 형태의 OpenAI Batch ID가 발급되며 `batch_jobs.json` 메타파일에 자동 기록됩니다.*

---

### 3단계: 번역 작업 진행 상태 확인 (대기)
제출한 Batch 작업들이 OpenAI 서버에서 완료될 때까지 기다리면서 수시로 상태를 확인합니다. (보통 5분 ~ 수십 분 정도 걸릴 수 있습니다.)

```bash
python scripts/src/translate_skeleton_batch.py \
    --check \
    --batch_dir data/batch_requests/skeleton_translation
```
*모든 언어의 Status가 `completed`로 바뀔 때까지 대기합니다.*

---

### 4단계: 완료된 번역 결과 다운로드
상태 확인에서 완료(`completed`)가 확인되면 번역 결과물을 내 컴퓨터에 다운로드합니다.

```bash
python scripts/src/translate_skeleton_batch.py \
    --download \
    --batch_dir data/batch_requests/skeleton_translation \
    --download_dir data/batch_requests/skeleton_translation/results
```
*`data/batch_requests/skeleton_translation/results/` 디렉토리 안에 `batch_result_zh.jsonl`, `batch_result_es.jsonl` 등의 번역 완료 파일들이 내려받아집니다.*

---

### 5단계: 번역 결과물 결합 (후처리)
다운로드받은 다국어 번역 skeleton을 원본 영문 skeleton 데이터셋 포맷에 맞추어 병합(Stitch)합니다. 이 작업은 5개 언어에 대하여 각각 실행해 줍니다.

```bash
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
```
*실행하면 각 언어별로 병합 성공률과 실패 건에 대한 영어 fallback 처리가 화면에 리포트되며 완료 폴더에 데이터셋이 안착됩니다.*

---

### 6단계: 번역된 Skeleton으로 고속 모델 평가 실행
마지막으로 번역된 skeleton들을 사용해서 일괄적으로 Solving (Turn 2) 평가를 수행합니다.

```bash
bash scripts/eval/eval_with_translated_skeleton.sh
```
*이 스크립트는 내부적으로 4개의 GPU를 병렬로 사용하여 번역본 파일이 발견되는 각 언어(`en`, `zh`, `ru`, `es`, `ko`, `th`)들에 대해 순차적으로 vLLM 추론 평가를 수행하고 결과를 지정된 파일에 자동으로 저장합니다.*