import argparse
import json
import os
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
        "ta": "சரி,\n",  # Tamil
        "kn": "ಸರಿ,\n",  # Kannada
        "si": "හොඳයි,\n",  # Sinhala
        "gu": "સારું,\n",  # Gujarati
        "ne": "हुन्छ,\n",  # Nepali
        "ml": "ശരി,\n",  # Malayalam
        "mr": "ठीक,\n",  # Marathi
        "or": "ଠିକ୍,\n",  # Odia
        "pa": "ਠੀਕ,\n",  # Punjabi
        "my": "ကောင်းပြီ,\n",  # Burmese
        "km": "ល្អ,\n",  # Khmer
        "lo": "ໂດຍ,\n",  # Lao
        "jv": "Inggih,\n",  # Javanese
        "su": "Leres,\n",  # Sundanese
        "ceb": "Sige,\n",  # Cebuano
        "uz": "Yaxshi,\n",  # Uzbek
        "ky": "Жакшы,\n",  # Kyrgyz
        "kk": "Жақсы,\n",  # Kazakh
        "tg": "Хуб,\n",  # Tajik
        "ug": "ياخشى,\n",  # Uyghur
        "ka": "კარგი,\n",  # Georgian
        "hy": "Լավ,\n",  # Armenian
        "am": "እሺ,\n",  # Amharic
        "yo": "O da,\n",  # Yoruba
        "so": "Hagaag,\n",  # Somali
        "mg": "Eny,\n",  # Malagasy
        "eu": "Ados,\n",  # Basque
        "mt": "Tajjeb,\n",  # Maltese
        "ku": "Baş e,\n",  # Kurdish
        "ps": "ښه,\n",  # Pashto
        "sd": "ٺيڪ,\n",  # Sindhi
        "gn": "Oĩma,\n",  # Guarani
        "qu": "Allin,\n",  # Quechua
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Rollouts from Existing Skeletons"
    )
    parser.add_argument("--skeleton_file", type=str, required=True, help="Path to the JSONL file with pre-extracted skeletons")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_k", type=float, default=-1)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--gpu_mem", type=float, default=0.92)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--rollout", type=int, default=10)
    parser.add_argument("--translate_cot", action="store_true", help="Generate CoT in English")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"📖 Reading skeleton file: {args.skeleton_file}")
    records = []
    with open(args.skeleton_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    print(f"✅ Loaded {len(records)} records.")

    tp = 4 if "qwen" in args.model.lower() and "7b" in args.model.lower() else args.tp

    print("🔧 Loading LLM...")
    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": tp,
        "gpu_memory_utilization": args.gpu_mem,
        "dtype": args.dtype,
        "hf_token": HF_TOKEN,
    }

    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    gen_sp = SamplingParams(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        n=args.rollout,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"🚀 Processing and generating {args.rollout} rollouts...")
    lora_adapter = args.lora

    with open(args.output, "w", encoding="utf-8") as f_out:
        for batch in tqdm(
            chunks(records, args.batch),
            total=(len(records) // args.batch + 1),
        ):
            sol_prompts = []
            sol_prompt_mapping = []

            for batch_idx, item in enumerate(batch):
                skeleton_list = item.get("skeleton", [])
                sk = skeleton_list[0] if skeleton_list else ""
                if not sk:
                    continue
                force_en = args.translate_cot and item.get("question_language") != "en"
                sol_prompts.append(
                    build_solver_prompt(
                        tok,
                        item["prompt"],
                        sk,
                        item["question_language"],
                        force_english=force_en,
                    )
                )
                sol_prompt_mapping.append(batch_idx)

            responses_list = [[] for _ in batch]

            if sol_prompts:
                if lora_adapter:
                    sol_outputs = llm.generate(
                        sol_prompts,
                        gen_sp,
                        use_tqdm=False,
                        lora_request=LoRARequest("adapter", 1, lora_adapter),
                    )
                else:
                    sol_outputs = llm.generate(sol_prompts, gen_sp, use_tqdm=False)

                for mapped_batch_idx, out in zip(sol_prompt_mapping, sol_outputs):
                    if out.outputs:
                        responses_list[mapped_batch_idx] = [
                            o.text.strip() for o in out.outputs
                        ]

            for i, item in enumerate(batch):
                item["responses"] = responses_list[i] if responses_list[i] else [""]
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"🎉 Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
