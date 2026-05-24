import argparse
import json
import os
import random
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
HF_TOKEN = os.getenv("HF_TOKEN")


# ═══════════════════════════════════════════════════════════════════════════════
#                              SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT_SKELETON = r"""Your Role:
You generate minimal, high-level reasoning skeletons for math and science problems.

Language Requirement:
Always respond in English, regardless of the language used in the user query.

Goal:
Provide only the *essential structural characteristics* of the problem. You must describe WHAT the problem structure is, never HOW to solve it.

Output Format: Provide exactly two sections below.

---

1. **Problem Structure**

Identify and characterize:
- **Objects & Variables**: Types of entities (quantities, sets, functions, geometric figures), separating constants/parameters from variables. Distinguish knowns vs. unknowns.
- **Relationships**: The nature of connections between entities (e.g., proportional, inverse, functional dependency, spatial). Describe the *type* of relationship, not the derivation.
- **Constraints**: Conditions defining the problem space (equations, inequalities, domain restrictions, boundary conditions, conservation laws).
- **Problem Nature**: (Only if relevant) Briefly note the structural type (e.g., Optimization, Combinatorial, Inverse problem).

*Constraints for this section:*
- Must be problem-specific, not generic.
- Must NOT include solution steps, strategies, formulas, or numerical manipulations.

2. **Key Concepts / Tools**

List relevant mathematical or scientific principles using standard terminology (e.g., "Pythagorean Theorem", "Conservation of Energy", "Bayes' Theorem").

*Constraints for this section:*
- Use established concept names only.
- Do NOT explain how to apply them or list formulas.
"""

SYSTEM_PROMPT_COT = r"""You are a math problem solver. Solve the given problem.

At the end, clearly state the final answer in the following format:
\boxed{final answer}
"""

SYSTEM_PROMPT_TRANSLATOR = r"""You are a professional academic translator specializing in mathematical content.

### TASK:
Translate the given math problem into clear, natural, and precise English, regardless of the source language.

### INSTRUCTIONS:
1. You SHOULD NOT solve the problem and translate only the given question — do not include any additional commentary.
2. Preserve all mathematical symbols, notations, formatting, and existing choices exactly as presented.
3. Use fluent, natural English that aligns with academic standards for math problems.
4. Ensure the translation conveys the meaning and context accurately.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#                           DATASET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# DATASET_CONFIGS = {
#     "polymath": {
#         "type": "polymath", 
#         "col_map": {"question": "question", "answer": "answer", "id": "id"},
#     },
#     "aime25": {
#         "type": "aime25",
#         "col_map": {"question": "question", "answer": "answer", "id": "id"},
#     },
#     "math500": {
#         "type": "math500",
#         "col_map": {"problem": "question", "answer": "answer", "unique_id": "id", "level": "difficulty"},
#     },
#     "mgsm": {
#         "type": "mgsm",
#         "col_map": {"question": "question", "answer_number": "answer"},
#     },
#     "default": {
#         "type": "default",
#         "col_map": {"question": "question", "answer": "answer", "id": "id"},
#     }
# }

DATASET_CONFIGS = {
    "polymath": {
        "type": "polymath", 
        "col_map": {"translated_en": "question", "answer": "answer", "id": "id"},
    },
    "aime25": {
        "type": "aime25",
        "col_map": {"translated_en": "question", "answer": "answer", "id": "id"},
    },
    "math500": {
        "type": "math500",
        "col_map": {"translated_en": "question", "answer": "answer", "unique_id": "id", "level": "difficulty"},
    },
    "mgsm": {
        "type": "mgsm",
        "col_map": {"translated_en": "question", "answer_number": "answer"},
    },
    "default": {
        "type": "default",
        "col_map": {"question": "question", "answer": "answer", "id": "id"},
    }
}



# ═══════════════════════════════════════════════════════════════════════════════
#                              ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined Evaluation Benchmark (Skeleton Multi-turn / CoT) with Translation Options"
    )
    
    # Method selection
    parser.add_argument("--method", type=str, required=True, 
                        choices=["skeleton_multiturn", "cot"],
                        help="Evaluation method")
    
    # Translation options
    parser.add_argument("--translate_q", action="store_true",
                        help="Translate question to English before solving (5-shot)")
    parser.add_argument("--translate_cot", action="store_true",
                        help="Generate CoT in English, then translate to target language (5-shot)")
    
    # Model/Inference args
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_k", type=float, default=-1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_mem", type=float, default=0.8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--rollout", type=int, default=1)
    
    parser.add_argument("--n_shot", type=int, default=0, help="Number of few-shot examples for solving")
    
    # Task Params
    parser.add_argument("--langs", type=str, default="en,es,zh,ko,th,sw,te")
    parser.add_argument("--difficulties", type=str, default="top,high,medium,low")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # CoT specific
    parser.add_argument("--modes", type=str, default="all")
    parser.add_argument("--reasoning_model", action="store_true")
    
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#                              HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataset_config(dataset_name: str) -> dict:
    name = dataset_name.lower()
    if "polymath" in name: return DATASET_CONFIGS["polymath"]
    if "aime" in name: return DATASET_CONFIGS["aime25"]
    if "math-500" in name: return DATASET_CONFIGS["math500"]
    if "mgsm" in name: return DATASET_CONFIGS["mgsm"]
    return DATASET_CONFIGS["default"]


def normalize_dataset(ds, config: dict):
    rename_map = {orig: tgt for orig, tgt in config["col_map"].items() if orig in ds.column_names}
    if rename_map:
        ds = ds.rename_columns(rename_map)
    if "id" not in ds.column_names:
        ds = ds.add_column("id", list(range(len(ds))))
    return ds


def sample_dataset(ds, ratio: float, seed: int):
    if ratio >= 1.0:
        return ds
    n = len(ds)
    k = max(1, int(n * ratio))
    random.seed(seed)
    indices = random.sample(range(n), k)
    return ds.select(indices)


def get_qa_text(row: dict, lang: str) -> str:
    lang_col_map = {
        "en": "question_en", "es": "question_es", "zh": "question_zh",
        "ko": "question_ko", "th": "question_th", "sw": "question_sw", "te": "question_te"
    }
    lang_col = lang_col_map.get(lang)
    if lang_col and row.get(lang_col):
        return row[lang_col]
    return row.get("question", "")


def chunks(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_lang_name(lang_code: str) -> str:
    names = {
        "en": "English", "es": "Spanish", "zh": "Chinese",
        "ko": "Korean", "th": "Thai", "sw": "Swahili", "te": "Telugu"
    }
    return names.get(lang_code, lang_code)


# ═══════════════════════════════════════════════════════════════════════════════
#                           TRANSLATION PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_translation_prompt(tok, text: str, src_lang: str, tgt_lang: str) -> str:
    """Build a 5-shot translation prompt."""
    fewshot_key = f"{src_lang}_to_{tgt_lang}"
    fewshots = TRANSLATION_FEWSHOTS.get(fewshot_key, [])
    
    src_name = get_lang_name(src_lang)
    tgt_name = get_lang_name(tgt_lang)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TRANSLATOR}]
    
    # Add 5 few-shot examples
    for shot in fewshots[:5]:
        messages.append({"role": "user", "content": f"Translate from {src_name} to {tgt_name}:\n{shot['src']}"})
        messages.append({"role": "assistant", "content": shot['tgt']})
    
    # Add the actual translation request
    messages.append({"role": "user", "content": f"Translate from {src_name} to {tgt_name}:\n{text}"})
    
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ═══════════════════════════════════════════════════════════════════════════════
#                        SKELETON MULTI-TURN PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

SKELETON_FEWSHOTS = {
    "en": [
        {
            "q": "A triangle has sides of length 5 cm and 12 cm, with a right angle between them. What is the length of the third side?",
            "a": "**Problem Structure**\n"
                 "- Geometric figure: right triangle with three sides\n"
                 "- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n"
                 "- Unknown: third side (hypotenuse) connecting the two legs\n"
                 "- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Pythagorean theorem\n"
                 "- Right triangle geometry"
        },
        {
            "q": "A committee needs to pick 2 members from a group of 5 candidates (Alice, Bob, Charlie, Dave, Eve). However, Alice and Bob cannot be on the committee together. How many ways can the committee be formed?",
            "a": "**Problem Structure**\n"
                 "- Discrete selection problem: choosing k=2 elements from n=5 candidates\n"
                 "- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n"
                 "- Unknown: number of valid selection configurations\n"
                 "- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n"
                 "- Solution space partitioned into valid and invalid subsets by constraint\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Combinatorial counting (combinations)\n"
                 "- Constraint satisfaction\n"
                 "- Complementary counting"
        },
        {
            "q": "Machine A runs for 8 hours producing 100 units/hour. Machine B runs for 4 hours, rests for 1 hour, then runs for 3 hours at 120 units/hour. What is the total production?",
            "a": "**Problem Structure**\n"
                 "- Two independent production processes with different temporal patterns\n"
                 "- Variables: production rates (constant per machine-segment), time durations, total output\n"
                 "- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n"
                 "- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour) separated by idle period\n"
                 "- Relationship: total output = sum of (rate × active duration) across all segments\n"
                 "- Known: all rates and time durations; Unknown: aggregate production quantity\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Rate-time-quantity relationships\n"
                 "- Piecewise continuous processes\n"
                 "- Linear aggregation"
        }
    ],
    "ko": [
        {
            "q": "한 변의 길이가 5cm이고 다른 한 변의 길이가 12cm인 직각삼각형이 있습니다. 세 번째 변의 길이는 얼마입니까?",
            "a": "**Problem Structure**\n"
                 "- Geometric figure: right triangle with three sides\n"
                 "- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n"
                 "- Unknown: third side (hypotenuse) connecting the two legs\n"
                 "- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Pythagorean theorem\n"
                 "- Right triangle geometry"
        },
        {
            "q": "5명의 후보(Alice, Bob, Charlie, Dave, Eve) 중에서 2명의 위원을 선출해야 합니다. 그러나 Alice와 Bob은 함께 위원이 될 수 없습니다. 위원회를 구성하는 방법은 몇 가지입니까?",
            "a": "**Problem Structure**\n"
                 "- Discrete selection problem: choosing k=2 elements from n=5 candidates\n"
                 "- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n"
                 "- Unknown: number of valid selection configurations\n"
                 "- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n"
                 "- Solution space partitioned into valid and invalid subsets by constraint\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Combinatorial counting (combinations)\n"
                 "- Constraint satisfaction\n"
                 "- Complementary counting"
        },
        {
            "q": "기계 A는 8시간 동안 시간당 100개를 생산합니다. 기계 B는 4시간 가동 후 1시간 휴식하고, 그 후 3시간 동안 시간당 120개를 생산합니다. 총 생산량은 얼마입니까?",
            "a": "**Problem Structure**\n"
                 "- Two independent production processes with different temporal patterns\n"
                 "- Variables: production rates (constant per machine-segment), time durations, total output\n"
                 "- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n"
                 "- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour) separated by idle period\n"
                 "- Relationship: total output = sum of (rate × active duration) across all segments\n"
                 "- Known: all rates and time durations; Unknown: aggregate production quantity\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Rate-time-quantity relationships\n"
                 "- Piecewise continuous processes\n"
                 "- Linear aggregation"
        }
    ],
    "es": [
        {
            "q": "Un triángulo tiene lados de longitud 5 cm y 12 cm, con un ángulo recto entre ellos. ¿Cuál es la longitud del tercer lado?",
            "a": "**Problem Structure**\n"
                 "- Geometric figure: right triangle with three sides\n"
                 "- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n"
                 "- Unknown: third side (hypotenuse) connecting the two legs\n"
                 "- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Pythagorean theorem\n"
                 "- Right triangle geometry"
        },
        {
            "q": "Un comité necesita elegir 2 miembros de un grupo de 5 candidatos (Alice, Bob, Charlie, Dave, Eve). Sin embargo, Alice y Bob no pueden estar juntos en el comité. ¿De cuántas maneras se puede formar el comité?",
            "a": "**Problem Structure**\n"
                 "- Discrete selection problem: choosing k=2 elements from n=5 candidates\n"
                 "- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n"
                 "- Unknown: number of valid selection configurations\n"
                 "- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n"
                 "- Solution space partitioned into valid and invalid subsets by constraint\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Combinatorial counting (combinations)\n"
                 "- Constraint satisfaction\n"
                 "- Complementary counting"
        },
        {
            "q": "La máquina A funciona durante 8 horas produciendo 100 unidades/hora. La máquina B funciona durante 4 horas, descansa 1 hora y luego funciona 3 horas a 120 unidades/hora. ¿Cuál es la producción total?",
            "a": "**Problem Structure**\n"
                 "- Two independent production processes with different temporal patterns\n"
                 "- Variables: production rates (constant per machine-segment), time durations, total output\n"
                 "- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n"
                 "- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour) separated by idle period\n"
                 "- Relationship: total output = sum of (rate × active duration) across all segments\n"
                 "- Known: all rates and time durations; Unknown: aggregate production quantity\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Rate-time-quantity relationships\n"
                 "- Piecewise continuous processes\n"
                 "- Linear aggregation"
        }
    ],
    "zh": [
        {
            "q": "一个三角形的两条边长分别为5厘米和12厘米，它们之间夹着一个直角。第三条边的长度是多少？",
            "a": "**Problem Structure**\n"
                 "- Geometric figure: right triangle with three sides\n"
                 "- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n"
                 "- Unknown: third side (hypotenuse) connecting the two legs\n"
                 "- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Pythagorean theorem\n"
                 "- Right triangle geometry"
        },
        {
            "q": "一个委员会需要从5名候选人（Alice, Bob, Charlie, Dave, Eve）中选出2名成员。但是，Alice和Bob不能同时在委员会中。委员会可以有多少种组成方式？",
            "a": "**Problem Structure**\n"
                 "- Discrete selection problem: choosing k=2 elements from n=5 candidates\n"
                 "- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n"
                 "- Unknown: number of valid selection configurations\n"
                 "- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n"
                 "- Solution space partitioned into valid and invalid subsets by constraint\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Combinatorial counting (combinations)\n"
                 "- Constraint satisfaction\n"
                 "- Complementary counting"
        },
        {
            "q": "机器A运行8小时，每小时生产100个单位。机器B运行4小时，休息1小时，然后以每小时120个单位的速度运行3小时。总产量是多少？",
            "a": "**Problem Structure**\n"
                 "- Two independent production processes with different temporal patterns\n"
                 "- Variables: production rates (constant per machine-segment), time durations, total output\n"
                 "- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n"
                 "- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour) separated by idle period\n"
                 "- Relationship: total output = sum of (rate × active duration) across all segments\n"
                 "- Known: all rates and time durations; Unknown: aggregate production quantity\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Rate-time-quantity relationships\n"
                 "- Piecewise continuous processes\n"
                 "- Linear aggregation"
        }
    ],
    "th": [
        {
            "q": "สามเหลี่ยมรูปหนึ่งมีด้านยาว 5 ซม. และ 12 ซม. โดยมีมุมฉากอยู่ระหว่างด้านทั้งสอง ด้านที่สามยาวเท่าไร?",
            "a": "**Problem Structure**\n"
                 "- Geometric figure: right triangle with three sides\n"
                 "- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n"
                 "- Unknown: third side (hypotenuse) connecting the two legs\n"
                 "- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Pythagorean theorem\n"
                 "- Right triangle geometry"
        },
        {
            "q": "คณะกรรมการต้องการเลือกสมาชิก 2 คนจากกลุ่มผู้สมัคร 5 คน (Alice, Bob, Charlie, Dave, Eve) อย่างไรก็ตาม Alice และ Bob ไม่สามารถอยู่ในคณะกรรมการร่วมกันได้ จะมีกี่วิธีในการจัดตั้งคณะกรรมการ?",
            "a": "**Problem Structure**\n"
                 "- Discrete selection problem: choosing k=2 elements from n=5 candidates\n"
                 "- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n"
                 "- Unknown: number of valid selection configurations\n"
                 "- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n"
                 "- Solution space partitioned into valid and invalid subsets by constraint\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Combinatorial counting (combinations)\n"
                 "- Constraint satisfaction\n"
                 "- Complementary counting"
        },
        {
            "q": "เครื่องจักร A ทำงาน 8 ชั่วโมง ผลิตได้ 100 หน่วย/ชั่วโมง เครื่องจักร B ทำงาน 4 ชั่วโมง พัก 1 ชั่วโมง จากนั้นทำงานต่อ 3 ชั่วโมงที่ 120 หน่วย/ชั่วโมง ผลผลิตรวมทั้งหมดคือเท่าไร?",
            "a": "**Problem Structure**\n"
                 "- Two independent production processes with different temporal patterns\n"
                 "- Variables: production rates (constant per machine-segment), time durations, total output\n"
                 "- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n"
                 "- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour) separated by idle period\n"
                 "- Relationship: total output = sum of (rate × active duration) across all segments\n"
                 "- Known: all rates and time durations; Unknown: aggregate production quantity\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Rate-time-quantity relationships\n"
                 "- Piecewise continuous processes\n"
                 "- Linear aggregation"
        }
    ],
    "sw": [
        {
            "q": "Pembetatu ina pande zenye urefu wa sentimita 5 na 12, na kuna pembe mraba kati yake. Je, urefu wa upande wa tatu ni nini?",
            "a": "**Problem Structure**\n"
                 "- Geometric figure: right triangle with three sides\n"
                 "- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n"
                 "- Unknown: third side (hypotenuse) connecting the two legs\n"
                 "- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Pythagorean theorem\n"
                 "- Right triangle geometry"
        },
        {
            "q": "Kamati inahitaji kuchagua wanachama 2 kutoka kundi la wagombea 5 (Alice, Bob, Charlie, Dave, Eve). Hata hivyo, Alice na Bob hawawezi kuwa kwenye kamati pamoja. Kamati inaweza kuundwa kwa njia ngapi?",
            "a": "**Problem Structure**\n"
                 "- Discrete selection problem: choosing k=2 elements from n=5 candidates\n"
                 "- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n"
                 "- Unknown: number of valid selection configurations\n"
                 "- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n"
                 "- Solution space partitioned into valid and invalid subsets by constraint\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Combinatorial counting (combinations)\n"
                 "- Constraint satisfaction\n"
                 "- Complementary counting"
        },
        {
            "q": "Mashine A inafanya kazi kwa masaa 8 ikizalisha vitengo 100/saa. Mashine B inafanya kazi kwa masaa 4, inapumzika saa 1, kisha inafanya kazi kwa masaa 3 kwa vitengo 120/saa. Uzalishaji wote ni upi?",
            "a": "**Problem Structure**\n"
                 "- Two independent production processes with different temporal patterns\n"
                 "- Variables: production rates (constant per machine-segment), time durations, total output\n"
                 "- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n"
                 "- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour) separated by idle period\n"
                 "- Relationship: total output = sum of (rate × active duration) across all segments\n"
                 "- Known: all rates and time durations; Unknown: aggregate production quantity\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Rate-time-quantity relationships\n"
                 "- Piecewise continuous processes\n"
                 "- Linear aggregation"
        }
    ],
    "te": [
        {
            "q": "ఒక త్రిభుజం 5 సెం.మీ మరియు 12 సెం.మీ పొడవు గల భుజాలను కలిగి ఉంది, వాటి మధ్య లంబకోణం ఉంది. మూడవ భుజం పొడవు ఎంత?",
            "a": "**Problem Structure**\n"
                 "- Geometric figure: right triangle with three sides\n"
                 "- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n"
                 "- Unknown: third side (hypotenuse) connecting the two legs\n"
                 "- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Pythagorean theorem\n"
                 "- Right triangle geometry"
        },
        {
            "q": "5 అభ్యర్థుల (Alice, Bob, Charlie, Dave, Eve) సమూహం నుండి ఒక కమిటీ 2 సభ్యులను ఎన్నుకోవాలి. అయితే, Alice మరియు Bob కలిసి కమిటీలో ఉండలేరు. కమిటీని ఎన్ని విధాలుగా ఏర్పాటు చేయవచ్చు?",
            "a": "**Problem Structure**\n"
                 "- Discrete selection problem: choosing k=2 elements from n=5 candidates\n"
                 "- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n"
                 "- Unknown: number of valid selection configurations\n"
                 "- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n"
                 "- Solution space partitioned into valid and invalid subsets by constraint\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Combinatorial counting (combinations)\n"
                 "- Constraint satisfaction\n"
                 "- Complementary counting"
        },
        {
            "q": "మెషిన్ A 8 గంటల పాటు గంటకు 100 యూనిట్లు ఉత్పత్తి చేస్తుంది. మెషిన్ B 4 గంటలు పనిచేసి, 1 గంట విశ్రాంతి తీసుకుని, తర్వాత 3 గంటల పాటు గంటకు 120 యూనిట్ల చొప్పున పనిచేస్తుంది. మొత్తం ఉత్పత్తి ఎంత?",
            "a": "**Problem Structure**\n"
                 "- Two independent production processes with different temporal patterns\n"
                 "- Variables: production rates (constant per machine-segment), time durations, total output\n"
                 "- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n"
                 "- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour) separated by idle period\n"
                 "- Relationship: total output = sum of (rate × active duration) across all segments\n"
                 "- Known: all rates and time durations; Unknown: aggregate production quantity\n\n"
                 "**Key Concepts / Tools**\n"
                 "- Rate-time-quantity relationships\n"
                 "- Piecewise continuous processes\n"
                 "- Linear aggregation"
        }
    ]
}

def build_skeleton_prompt(tok, lang: str, question: str) -> str:
    fewshots = SKELETON_FEWSHOTS.get(lang, SKELETON_FEWSHOTS["en"])
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT_SKELETON}]
    for shot in fewshots:
        messages.append({"role": "user", "content": shot["q"]})
        messages.append({"role": "assistant", "content": shot["a"]})
    messages.append({"role": "user", "content": f"Question: {question}\n\nYou must respond in English."})
    
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_solver_prompt(tok, question: str, skeleton: str, lang: str,
                        few_shots: list = None, force_english: bool = False) -> str:
    """Build solver prompt for skeleton-based solving."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT_COT}]

    if few_shots:
        for shot in few_shots:
            messages.append({"role": "user", "content": shot["question"]})
            messages.append({"role": "assistant", "content": shot["answer"]})

    # Determine language for reasoning
    reason_lang = "en" if force_english else lang

    lang_triggers = {
        "ko": "좋습니다,\n", "sw": "Sawa,\n", "es": "Bien,\n",
        "zh": "好的，\n", "th": "ดี\n", "te": "సరే,\n", "en": ""
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

    user_prompt = f"Question: {question}\n\nReasoning Skeleton:\n{skeleton}\n\n{lang_hint}"
    messages.append({"role": "user", "content": user_prompt})

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + start_trigger


# ═══════════════════════════════════════════════════════════════════════════════
#                           COT PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_cot_prompt(tok, prompt: str, few_shots: list, reasoning_lang: str,
                     reasoning_model: bool, force_english: bool = False) -> str:
    messages = []

    if not reasoning_model:
        messages.append({"role": "system", "content": SYSTEM_PROMPT_COT})

    if few_shots:
        for shot in few_shots:
            messages.append({"role": "user", "content": shot["question"]})
            messages.append({"role": "assistant", "content": shot["answer"]})

    messages.append({"role": "user", "content": prompt})

    # Determine actual reasoning language
    actual_lang = "en" if force_english else reasoning_lang

    lang_triggers_cot = {
        "ko": "좋습니다,", "sw": "Sawa,", "es": "Bien,",
        "zh": "好的，", "th": "ดี", "te": "సరే,", "en": ""
    }

    lang_name = get_lang_name(actual_lang)
    if actual_lang != "en":
        cot_hint = f"Let's think step by step. Respond in {lang_name}."
        forcing = lang_triggers_cot.get(actual_lang, "")
    else:
        cot_hint = "Let's think step by step."
        forcing = ""

    messages[-1]["content"] = f"{prompt}\n\n{cot_hint}"

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + forcing
    return text


def parse_mode(mode_str: str):
    s0 = mode_str.strip().replace("_", "-")
    parts = s0.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid mode: {mode_str}")
    q_lang, r_lang = parts[0], parts[1]
    s_flag = "-".join(parts[2:]).lower()
    skeleton = s_flag in {"skeleton", "ske"}
    return q_lang, r_lang, skeleton


def normalize_modes(modes_raw: str) -> list:
    if modes_raw.strip().lower() == "all":
        # All 7 languages: en, es, zh, ko, th, sw, te
        all_langs = ["en", "es", "zh", "ko", "th", "sw", "te"]
        modes = []
        for lang in all_langs:
            modes.append(f"{lang}-{lang}-nonskeleton")  # native reasoning
            if lang != "en":
                modes.append(f"{lang}-en-nonskeleton")  # english reasoning
                modes.append(f"en-{lang}-nonskeleton")  # english question, native reasoning
        return modes
    normed = []
    for m in modes_raw.split(","):
        m = m.strip()
        if not m:
            continue
        q, r, ske = parse_mode(m)
        ske_str = "skeleton" if ske else "nonskeleton"
        normed.append(f"{q}-{r}-{ske_str}")
    return list(set(normed))


# ═══════════════════════════════════════════════════════════════════════════════
#                           MAIN EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_skeleton_multiturn(args, llm, tok, config, gen_sp, sk_sp, trans_sp, load_targets, langs):
    """Skeleton-based Multi-turn evaluation with optional translation."""
    global_idx = 0
    lora_adapter = args.lora
    
    with open(args.output, "w", encoding="utf-8") as f_out:
        for target in load_targets:
            curr_lang = target["lang"]
            curr_split = target["split"]
            curr_name = target["name"]
            
            print(f"\n📥 Loading [{args.dataset}] Name={curr_name}, Split={curr_split} ...")
            
            try:
                ds = load_dataset(args.dataset, name=curr_name, split=curr_split) if curr_name else load_dataset(args.dataset, split=curr_split)
                ratio = 1 if config["type"] == "aime25" else args.sample_ratio
                ds = normalize_dataset(ds, config)
                ds = sample_dataset(ds, ratio, args.seed)
                print(f"✅ Loaded {len(ds)} samples.")
                
                # Build task queue
                task_queue = []
                target_langs = [curr_lang] if curr_lang else ["en"]
                
                for i, row in enumerate(ds):
                    unique_id = row.get("id", i)
                    answer = row.get("answer", "")
                    difficulty = row.get("difficulty", curr_split)
                    
                    for q_lang in target_langs:
                        if q_lang not in langs:
                            continue
                        question_text = get_qa_text(row, q_lang)
                        if not question_text or not question_text.strip():
                            continue
                        task_queue.append({
                            "unique_id": unique_id,
                            "question": question_text,
                            "answer": answer,
                            "difficulty": difficulty,
                            "lang": q_lang,
                        })
                
                if not task_queue:
                    print("⚠️ No valid tasks found.")
                    continue
                
                print(f"🚀 Processing {len(task_queue)} tasks (Skeleton Multi-turn)...")
                
                for batch in tqdm(chunks(task_queue, args.batch), total=(len(task_queue) // args.batch + 1)):
                    questions_to_use = [t["question"] for t in batch]
                    translated_questions = [None] * len(batch)
                    
                    # [Turn 1] Skeleton Generation
                    sk_prompts = [build_skeleton_prompt(tok, "en" if args.translate_q else t["lang"], q) 
                                  for t, q in zip(batch, questions_to_use)]
                    
                    if args.lora:
                        sk_outputs = llm.generate(sk_prompts, sk_sp, use_tqdm=False, 
                                                  lora_request=LoRARequest("adapter", 1, lora_adapter))
                    else:
                        sk_outputs = llm.generate(sk_prompts, sk_sp, use_tqdm=False)
                    
                    # For AIME25: sk_sp.n = 16, so each task gets 16 skeletons
                    # For others: sk_sp.n = 1, so each task gets 1 skeleton
                    is_aime25 = config["type"] == "aime25"
                    
                    if is_aime25:
                        # Each sk_output has 16 skeleton outputs
                        skeletons_list = [[o.text.strip() for o in out.outputs] if out.outputs else [""] for out in sk_outputs]
                    else:
                        # Each sk_output has 1 skeleton output
                        skeletons_list = [[out.outputs[0].text.strip()] if out.outputs else [""] for out in sk_outputs]
                    
                    # [Turn 2] Solver Generation
                    sol_prompts = []
                    sol_prompt_mapping = []  # (batch_idx, skeleton_idx)
                    
                    for batch_idx, (t, q, sks) in enumerate(zip(batch, questions_to_use, skeletons_list)):
                        for sk_idx, sk in enumerate(sks):
                            if not sk:
                                continue
                            force_en = args.translate_cot and t["lang"] != "en"
                            sol_prompts.append(build_solver_prompt(tok, q, sk, t["lang"], 
                                                                   force_english=force_en))
                            sol_prompt_mapping.append((batch_idx, sk_idx))
                    
                    sol_texts = [[] for _ in batch]
                    skeletons = [[] for _ in batch]  # Final skeletons storage
                    
                    if sol_prompts:
                        sol_outputs = llm.generate(sol_prompts, gen_sp, use_tqdm=False)
                        for (batch_idx, sk_idx), out in zip(sol_prompt_mapping, sol_outputs):
                            if out.outputs:
                                sol_texts[batch_idx].extend([o.text.strip() for o in out.outputs])
                                skeletons[batch_idx].append(skeletons_list[batch_idx][sk_idx])
                    
                    # Ensure empty lists have at least one empty string
                    for i in range(len(batch)):
                        if not sol_texts[i]:
                            sol_texts[i] = [""]
                        if not skeletons[i]:
                            skeletons[i] = [""]
                    
                    # [Optional Turn 3] Translate CoT back to target language
                    translated_cots = [None] * len(batch)
                    if args.translate_cot:
                        cot_trans_prompts = []
                        cot_trans_indices = []
                        for i, (t, sol) in enumerate(zip(batch, sol_texts)):
                            if t["lang"] != "en" and sol and sol[0]:
                                cot_trans_prompts.append(build_translation_prompt(tok, sol[0], "en", t["lang"]))
                                cot_trans_indices.append(i)
                        
                        if cot_trans_prompts:
                            cot_trans_outputs = llm.generate(cot_trans_prompts, trans_sp, use_tqdm=False)
                            for idx, out in zip(cot_trans_indices, cot_trans_outputs):
                                if out.outputs:
                                    translated_cots[idx] = out.outputs[0].text.strip()
                    
                    # Save results
                    for i, t in enumerate(batch):
                        record = {
                            "global_id": global_idx,
                            "original_id": t["unique_id"],
                            "prompt": t["question"],
                            "translated_question": translated_questions[i],
                            "skeleton": skeletons[i],
                            "input_prompt": sol_prompts[i],
                            "responses": sol_texts[i],
                            "translated_cot": translated_cots[i],
                            "question_language": t["lang"],
                            "difficulty": t["difficulty"],
                            "answer": t["answer"],
                            "method": "skeleton_multiturn",
                            "translate_q": args.translate_q,
                            "translate_cot": args.translate_cot,
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        global_idx += 1
                    f_out.flush()
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n🎉 Done! Saved to {args.output}")


def run_cot(args, llm, tok, config, gen_sp, trans_sp, load_targets, langs):
    """Direct CoT evaluation with optional translation."""
    mode_list = normalize_modes(args.modes)
    global_idx = 0
    
    with open(args.output, "w", encoding="utf-8") as f_out:
        for target in load_targets:
            curr_lang = target["lang"]
            curr_split = target["split"]
            curr_name = target["name"]
            
            print(f"\n📥 Loading [{args.dataset}] Name={curr_name}, Split={curr_split} ...")
            
            try:
                ds = load_dataset(args.dataset, name=curr_name, split=curr_split) if curr_name else load_dataset(args.dataset, split=curr_split)
                ratio = 1 if config["type"] == "aime25" else args.sample_ratio
                ds = normalize_dataset(ds, config)
                ds = sample_dataset(ds, ratio, args.seed)
                print(f"✅ Loaded {len(ds)} samples.")
                
                # Build tasks
                all_tasks = []
                for i, row in enumerate(ds):
                    unique_id = row.get("id", i)
                    answer = row.get("answer", None)
                    has_multilang = any([row.get(f"question_{l}") for l in ["en", "es", "zh", "ko", "th", "sw", "te"]])
                    
                    for mode in mode_list:
                        q_lang, r_lang, skeleton = parse_mode(mode)
                        question_text = get_qa_text(row, q_lang)
                        
                        if curr_lang and q_lang != curr_lang:
                            continue
                        if not question_text or not question_text.strip():
                            continue
                        
                        all_tasks.append({
                            "unique_id": unique_id,
                            "prompt": question_text,
                            "question_language": q_lang,
                            "reasoning_language": r_lang,
                            "mode": mode,
                            "difficulty": row.get("difficulty", curr_split),
                            "answer": answer,
                        })
                
                if not all_tasks:
                    print("⚠️ No tasks match. Skipping...")
                    continue
                
                print(f"🚀 Generating {len(all_tasks)} tasks...")
                
                for batch in tqdm(chunks(all_tasks, args.batch), total=(len(all_tasks) // args.batch + 1)):
                    questions_to_use = [t["prompt"] for t in batch]
                    translated_questions = [None] * len(batch)
                    
                    # [Optional] Translate question to English
                    if args.translate_q:
                        trans_prompts = []
                        trans_indices = []
                        for i, t in enumerate(batch):
                            if t["question_language"] != "en":
                                trans_prompts.append(build_translation_prompt(tok, t["prompt"], t["question_language"], "en"))
                                trans_indices.append(i)
                        
                        if trans_prompts:
                            trans_outputs = llm.generate(trans_prompts, trans_sp, use_tqdm=False)
                            for idx, out in zip(trans_indices, trans_outputs):
                                if out.outputs:
                                    translated_q = out.outputs[0].text.strip()
                                    translated_questions[idx] = translated_q
                                    questions_to_use[idx] = translated_q
                    
                    # Generate CoT
                    cot_prompts = []
                    for t, q in zip(batch, questions_to_use):
                        force_en = args.translate_cot and t["reasoning_language"] != "en"
                        effective_lang = "en" if force_en else t["reasoning_language"]
                        cot_prompts.append(build_cot_prompt(tok, q, [], effective_lang, args.reasoning_model, force_en))
                    
                    sol_outputs = llm.generate(cot_prompts, gen_sp, use_tqdm=False)
                    sol_texts = [[o.text.strip() for o in out.outputs] if out.outputs else [] for out in sol_outputs]
                    
                    # [Optional] Translate CoT back to target language
                    translated_cots = [None] * len(batch)
                    if args.translate_cot:
                        cot_trans_prompts = []
                        cot_trans_indices = []
                        for i, (t, sol) in enumerate(zip(batch, sol_texts)):
                            if t["reasoning_language"] != "en" and sol:
                                cot_trans_prompts.append(build_translation_prompt(tok, sol[0], "en", t["reasoning_language"]))
                                cot_trans_indices.append(i)
                        
                        if cot_trans_prompts:
                            cot_trans_outputs = llm.generate(cot_trans_prompts, trans_sp, use_tqdm=False)
                            for idx, out in zip(cot_trans_indices, cot_trans_outputs):
                                if out.outputs:
                                    translated_cots[idx] = out.outputs[0].text.strip()
                    
                    # Save
                    for i, t in enumerate(batch):
                        record = {
                            "global_id": global_idx,
                            "original_id": t["unique_id"],
                            "prompt": t["prompt"],
                            "translated_question": translated_questions[i],
                            "responses": sol_texts[i],
                            "translated_cot": translated_cots[i],
                            "question_language": t["question_language"],
                            "reasoning_language": t["reasoning_language"],
                            "mode": t["mode"],
                            "difficulty": t["difficulty"],
                            "answer": t["answer"],
                            "method": "cot",
                            "translate_q": args.translate_q,
                            "translate_cot": args.translate_cot,
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        global_idx += 1
                    f_out.flush()
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n🎉 Done! Saved to {args.output}")


# ═══════════════════════════════════════════════════════════════════════════════
#                                   MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    config = get_dataset_config(args.dataset)
    
    print(f"🛠️ Dataset Type: {config['type']}")
    print(f"🎯 Method: {args.method}")
    print(f"🔄 Translate Q: {args.translate_q} | Translate CoT: {args.translate_cot}")
    
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    
    tp = 4 if "qwen" in args.model.lower() and "7b" in args.model.lower() else args.tp
    
    # Load LLM
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
    
    eff_temp = 0.7 if args.rollout > 1 else args.temp
    eff_top_p = 0.95 if args.rollout > 1 else args.top_p
    
    # Sampling Parameters
    if config["type"] == "aime25":
        # For AIME25: generate 16 skeletons and 1 solution per skeleton
        gen_sp = SamplingParams(temperature=eff_temp, max_tokens=16384, n=args.rollout, top_p=eff_top_p)
        sk_sp = SamplingParams(temperature=eff_temp, max_tokens=args.max_tokens, n=16, top_p=eff_top_p)
    else:
        gen_sp = SamplingParams(
            temperature=eff_temp, 
            max_tokens=args.max_tokens, 
            top_k=args.top_k, 
            top_p=eff_top_p, 
            n=args.rollout
        )
        
        sk_sp = SamplingParams(
            temperature=0.0, 
            max_tokens=args.max_tokens, 
            top_k=args.top_k, 
            top_p=1.0
        )
    trans_sp = SamplingParams(temperature=0.0, max_tokens=4096, top_p=1.0)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Build load targets
    load_targets = []
    if config["type"] == "polymath":
        for l in langs:
            for d in diffs:
                load_targets.append({"lang": l, "split": d, "name": l})
    elif config["type"] in ["math500", "aime25"]:
        for l in langs:
            load_targets.append({"lang": l, "split": l, "name": None})
    elif config["type"] == "mgsm":
        for l in langs:
            load_targets.append({"lang": l, "split": "test", "name": l})
    else:
        for d in diffs:
            load_targets.append({"lang": "en", "split": d, "name": None})
    
    # Run evaluation
    if args.method == "skeleton_multiturn":
        run_skeleton_multiturn(args, llm, tok, config, gen_sp, sk_sp, trans_sp, load_targets, langs)
    elif args.method == "cot":
        run_cot(args, llm, tok, config, gen_sp, trans_sp, load_targets, langs)


if __name__ == "__main__":
    main()
