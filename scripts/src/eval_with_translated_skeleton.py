import argparse
import json
import os
import sys
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

HF_TOKEN = os.getenv("HF_TOKEN")

def get_lang_name(lang_code: str) -> str:
    names = {
        "ta": "Tamil",
        "kn": "Kannada",
        "my": "Burmese",
        "km": "Khmer",
        "am": "Amharic",
        "yo": "Yoruba",
        "si": "Sinhala",
        "gu": "Gujarati",
        "ne": "Nepali",
        "uz": "Uzbek",
        "ky": "Kyrgyz",
        "ceb": "Cebuano",
        "eu": "Basque",
        "gn": "Guarani",
        "hy": "Armenian",
        "jv": "Javanese",
        "ka": "Georgian",
        "kk": "Kazakh",
        "ku": "Kurdish",
        "lo": "Lao",
        "mg": "Malagasy",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mr": "Marathi",
        "mt": "Maltese",
        "or": "Odia",
        "pa": "Punjabi",
        "ps": "Pashto",
        "qu": "Quechua",
        "sd": "Sindhi",
        "so": "Somali",
        "su": "Sundanese",
        "tg": "Tajik",
        "ug": "Uyghur",
        "zh": "Chinese",
        "es": "Spanish",
        "ko": "Korean",
        "th": "Thai",
        "ru": "Russian",
        "en": "English"
    }
    return names.get(lang_code, lang_code)

def build_solver_prompt(
    tok,
    question: str,
    skeleton: str,
    lang: str,
    few_shots: list = None,
    force_english: bool = False,
) -> str:
    """Build solver prompt for skeleton-based solving."""
    messages = []

    if few_shots:
        for shot in few_shots:
            messages.append({"role": "user", "content": shot["question"]})
            messages.append({"role": "assistant", "content": shot["answer"]})

    reason_lang = "en" if force_english else lang

    lang_triggers = {
        # 인도 언어들
        "ta": "சரி,\n",  # Tamil
        "kn": "ಸರಿ,\n",  # Kannada
        "si": "හොඳයි,\n",  # Sinhala
        "gu": "સારું,\n",  # Gujarati
        "ne": "हुन्छ,\n",  # Nepali
        "ml": "ശരി,\n",  # Malayalam
        "mr": "ঠিক,\n",  # Marathi
        "or": "ଠିକ୍,\n",  # Odia
        "pa": "ਠੀਕ,\n",  # Punjabi
        # 동남아시아
        "my": "ကောင်းပြီ,\n",  # Burmese
        "km": "ល្អ,\n",  # Khmer
        "lo": "ໂດຍ,\n",  # Lao
        "jv": "Inggih,\n",  # Javanese
        "su": "Leres,\n",  # Sundanese
        "ceb": "Sige,\n",  # Cebuano
        # 중앙아시아 / 튀르크어
        "uz": "Yaxshi,\n",  # Uzbek
        "ky": "Жакшы,\n",  # Kyrgyz
        "kk": "Жақсы,\n",  # Kazakh
        "tg": "Хуб,\n",  # Tajik
        "ug": "ياخشى,\n",  # Uyghur
        # 캅카스 / 아르메니아
        "ka": "კარგი,\n",  # Georgian
        "hy": "Լավ,\n",  # Armenian
        # 아프리카
        "am": "እሺ,\n",  # Amharic
        "yo": "O da,\n",  # Yoruba
        "so": "Hagaag,\n",  # Somali
        "mg": "Eny,\n",  # Malagasy
        # 유럽
        "eu": "Ados,\n",  # Basque
        "mt": "Tajjeb,\n",  # Maltese
        # 중동
        "ku": "Baş e,\n",  # Kurdish
        "ps": "ښه,\n",  # Pashto
        "sd": "ٺيڪ,\n",  # Sindhi
        # 아메리카 원주민
        "gn": "Oĩma,\n",  # Guarani
        "qu": "Allin,\n",  # Quechua
        # 기타
        "mn": "За,\n",  # Mongolian
    }

    lang_name = get_lang_name(reason_lang)

    lang_hint = (
        "Instructions:\n"
        "1. Follow the Reasoning Skeleton above.\n"
        "2. Verify numbers against the Question text.\n"
        "3. Let's think step-by-step. Respond in {lang_name}.\n"
        "Final Answer Format: \\boxed{{answer}}"
    ).format(lang_name=lang_name)

    start_trigger = lang_triggers.get(reason_lang, "")

    user_prompt = (
        f"Question: {question}\n\nReasoning Skeleton:\n{skeleton}\n\n{lang_hint}"
    )
    messages.append({"role": "user", "content": user_prompt})

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + start_trigger

def chunks(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    parser = argparse.ArgumentParser(description="Evaluate with translated reasoning skeletons")
    
    # Input/Output
    parser.add_argument("--skeleton_file", type=str, required=True, help="Path to fully translated skeleton JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output JSONL")
    
    # Model config
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_k", type=float, default=-1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_mem", type=float, default=0.8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--rollout", type=int, default=1)
    parser.add_argument("--translate_cot", action="store_true", help="Generate CoT in English, then translate to target language")
    
    args = parser.parse_args()
    
    print(f"📖 Reading skeleton file: {args.skeleton_file}")
    if not os.path.exists(args.skeleton_file):
        print(f"❌ Skeleton file not found: {args.skeleton_file}")
        sys.exit(1)
        
    tasks = []
    with open(args.skeleton_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
                
    print(f"✅ Loaded {len(tasks)} tasks.")
    
    print("🔧 Loading LLM...")
    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tp,
        "gpu_memory_utilization": args.gpu_mem,
        "dtype": args.dtype,
        "hf_token": HF_TOKEN,
    }
    
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    
    eff_temp = 0.7 if args.rollout > 1 else args.temp
    eff_top_p = 0.8 if args.rollout > 1 else args.top_p

    gen_sp = SamplingParams(
        temperature=eff_temp,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=eff_top_p,
        n=args.rollout
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"🚀 Running evaluation for {len(tasks)} tasks ...")
    
    with open(args.output, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(
            chunks(tasks, args.batch),
            total=(len(tasks) // args.batch + 1)
        ):
            sol_prompts = []
            sol_prompt_mapping = []
            
            for idx, rec in enumerate(batch):
                question = rec.get("prompt")
                skeleton_list = rec.get("skeleton", [])
                if not skeleton_list or not skeleton_list[0]:
                    print(f"⚠️ Warning: Empty skeleton for global_id {rec.get('global_id')}")
                    skeleton_text = ""
                else:
                    skeleton_text = skeleton_list[0]
                    
                q_lang = rec.get("question_language", "en")
                force_en = args.translate_cot and q_lang != "en"
                
                prompt = build_solver_prompt(
                    tok=tok,
                    question=question,
                    skeleton=skeleton_text,
                    lang=q_lang,
                    force_english=force_en
                )
                
                sol_prompts.append(prompt)
                sol_prompt_mapping.append(idx)
                
            sol_texts = [[] for _ in batch]
            
            if sol_prompts:
                if args.lora:
                    sol_outputs = llm.generate(
                        sol_prompts,
                        gen_sp,
                        use_tqdm=False,
                        lora_request=LoRARequest("adapter", 1, args.lora)
                    )
                else:
                    sol_outputs = llm.generate(
                        sol_prompts,
                        gen_sp,
                        use_tqdm=False
                    )
                    
                for idx_in_prompts, out in enumerate(sol_outputs):
                    batch_idx = sol_prompt_mapping[idx_in_prompts]
                    if out.outputs:
                        sol_texts[batch_idx].extend([o.text.strip() for o in out.outputs])
                        
            # Save results
            for idx, rec in enumerate(batch):
                if not sol_texts[idx]:
                    sol_texts[idx] = [""]
                    
                # Prepare record maintaining key structure
                record = {
                    "global_id": rec.get("global_id"),
                    "original_id": rec.get("original_id"),
                    "prompt": rec.get("prompt"),
                    "skeleton": rec.get("skeleton"),
                    "skeleton_lang": rec.get("skeleton_lang"),
                    "responses": sol_texts[idx],
                    "question_language": rec.get("question_language"),
                    "difficulty": rec.get("difficulty"),
                    "answer": rec.get("answer"),
                    "method": rec.get("method"),
                }
                
                # Optional keys
                if "translated_question" in rec:
                    record["translated_question"] = rec["translated_question"]
                if "translated_cot" in rec:
                    record["translated_cot"] = rec["translated_cot"]
                    
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                
            f_out.flush()
            
    print(f"🎉 Evaluation finished! Saved to {args.output}")

if __name__ == "__main__":
    main()
