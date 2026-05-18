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
\\boxed{final answer}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#                           DATASET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

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
        "col_map": {"translated_en": "question", "answer": "answer", "id": "id"},
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
    parser.add_argument("--langs", type=str, default="ta,kn,my,km,am,yo,si,gu,ne,uz,ky,ceb,eu,gn,hy,jv,ka,kk,ku,lo,mg,ml,mn,mr,mt,or,pa,ps,qu,sd,so,su,tg,ug")
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
    rename_map = {}
    for orig, tgt in config["col_map"].items():
        if orig not in ds.column_names:
            continue
        
        # 만약 tgt 컬럼이 이미 존재한다면 → 먼저 제거
        if tgt in ds.column_names and tgt != orig:
            ds = ds.remove_columns([tgt])
        
        rename_map[orig] = tgt

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
        "ta": "question_ta", "kn": "question_kn", "my": "question_my",
        "km": "question_km", "am": "question_am", "yo": "question_yo",
        "si": "question_si", "gu": "question_gu", "ne": "question_ne",
        "uz": "question_uz", "ky": "question_ky",
        "ceb": "question_ceb", "eu": "question_eu", "gn": "question_gn",
        "hy": "question_hy", "jv": "question_jv", "ka": "question_ka",
        "kk": "question_kk", "ku": "question_ku", "lo": "question_lo",
        "mg": "question_mg", "ml": "question_ml", "mn": "question_mn",
        "mr": "question_mr", "mt": "question_mt", "or": "question_or",
        "pa": "question_pa", "ps": "question_ps", "qu": "question_qu",
        "sd": "question_sd", "so": "question_so", "su": "question_su",
        "tg": "question_tg", "ug": "question_ug"
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
        "ta": "Tamil", "kn": "Kannada", "my": "Burmese",
        "km": "Khmer", "am": "Amharic", "yo": "Yoruba",
        "si": "Sinhala", "gu": "Gujarati", "ne": "Nepali",
        "uz": "Uzbek", "ky": "Kyrgyz",
        "ceb": "Cebuano", "eu": "Basque", "gn": "Guarani",
        "hy": "Armenian", "jv": "Javanese", "ka": "Georgian",
        "kk": "Kazakh", "ku": "Kurdish", "lo": "Lao",
        "mg": "Malagasy", "ml": "Malayalam", "mn": "Mongolian",
        "mr": "Marathi", "mt": "Maltese", "or": "Odia",
        "pa": "Punjabi", "ps": "Pashto", "qu": "Quechua",
        "sd": "Sindhi", "so": "Somali", "su": "Sundanese",
        "tg": "Tajik", "ug": "Uyghur"
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

SKELETON_ANSWERS = [
    r"""**Problem Structure**
- Geometric figure: right triangle with three sides
- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint
- Unknown: third side (hypotenuse) connecting the two legs
- Relationship: side lengths satisfy distance relation enforced by right angle property

**Key Concepts / Tools**
- Pythagorean theorem
- Right triangle geometry""",

    r"""**Problem Structure**
- Discrete selection problem: choosing k=2 elements from n=5 candidates
- Known: total candidates (5), committee size (2), pairwise exclusion constraint
- Unknown: number of valid selection configurations
- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)
- Solution space partitioned into valid and invalid subsets by constraint

**Key Concepts / Tools**
- Combinatorial counting (combinations)
- Constraint satisfaction
- Complementary counting""",

    r"""**Problem Structure**
- Two independent production processes with different temporal patterns
- Variables: production rates (constant per machine-segment), time durations, total output
- Machine A: single continuous active interval (8 hours @ 100 units/hour)
- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour) separated by idle period
- Relationship: total output = sum of (rate × active duration) across all segments
- Known: all rates and time durations; Unknown: aggregate production quantity

**Key Concepts / Tools**
- Rate-time-quantity relationships
- Piecewise continuous processes
- Linear aggregation"""
]

SKELETON_QUESTIONS = {
    "en": [
        "A triangle has sides of length 5 cm and 12 cm, with a right angle between them. What is the length of the third side?",
        "A committee needs to pick 2 members from a group of 5 candidates (Alice, Bob, Charlie, Dave, Eve). However, Alice and Bob cannot be on the committee together. How many ways can the committee be formed?",
        "Machine A runs for 8 hours producing 100 units/hour. Machine B runs for 4 hours, rests for 1 hour, then runs for 3 hours at 120 units/hour. What is the total production?"
    ],
    "ta": [
        "ஒரு முக்கோணத்தின் பக்கங்கள் 5 செ.மீ மற்றும் 12 செ.மீ நீளம் கொண்டவை, அவற்றுக்கிடையே ஒரு செங்கோணம் உள்ளது. மூன்றாவது பக்கத்தின் நீளம் என்ன?",
        "5 பேர் கொண்ட குழுவில் (Alice, Bob, Charlie, Dave, Eve) இருந்து 2 உறுப்பினர்களைத் தேர்ந்தெடுக்க வேண்டும். ஆனால், Alice மற்றும் Bob இருவரும் ஒன்றாக குழுவில் இருக்க முடியாது. குழுவை எத்தனை வழிகளில் அமைக்கலாம்?",
        "இயந்திரம் A 8 மணி நேரம் இயங்கி மணிக்கு 100 அலகுகளை உற்பத்தி செய்கிறது. இயந்திரம் B 4 மணி நேரம் இயங்கி, 1 மணி நேரம் ஓய்வெடுத்து, பின்னர் 3 மணி நேரம் மணிக்கு 120 அலகுகள் வீதம் இயங்குகிறது. மொத்த உற்பத்தி என்ன?"
    ],
    "kn": [
        "ಒಂದು ತ್ರಿಕೋನವು 5 ಸೆಂ.ಮೀ ಮತ್ತು 12 ಸೆಂ.ಮೀ ಉದ್ದದ ಬಾಹುಗಳನ್ನು ಹೊಂದಿದ್ದು, ಅವುಗಳ ನಡುವೆ ಲಂಬಕೋನವಿದೆ. ಮೂರನೇ ಬಾಹುವಿನ ಉದ್ದ ಎಷ್ಟು?",
        "5 ಅಭ್ಯರ್ಥಿಗಳ ಗುಂಪಿನಿಂದ (Alice, Bob, Charlie, Dave, Eve) ಸಮಿತಿಯು 2 ಸದಸ್ಯರನ್ನು ಆಯ್ಕೆ ಮಾಡಬೇಕಾಗಿದೆ. ಆದರೆ, Alice ಮತ್ತು Bob ಇಬ್ಬರೂ ಒಟ್ಟಿಗೆ ಸಮಿತಿಯಲ್ಲಿ ಇರುವಂತಿಲ್ಲ. ಸಮಿತಿಯನ್ನು ಎಷ್ಟು ವಿಧಗಳಲ್ಲಿ ರಚಿಸಬಹುದು?",
        "ಮಷಿನ್ A ಗಂಟೆಗೆ 100 ಯುನಿಟ್‌ಗಳಂತೆ 8 ಗಂಟೆಗಳ ಕಾಲ ಚಲಿಸುತ್ತದೆ. ಮಷಿನ್ B 4 ಗಂಟೆಗಳ ಕಾಲ ಚಲಿಸುತ್ತದೆ, 1 ಗಂಟೆ ವಿಶ್ರಾಂತಿ ಪಡೆಯುತ್ತದೆ, ನಂತರ ಗಂಟೆಗೆ 120 ಯುನಿಟ್‌ಗಳಂತೆ 3 ಗಂಟೆಗಳ ಕಾಲ ಚಲಿಸುತ್ತದೆ. ಒಟ್ಟು ಉತ್ಪಾದನೆ ಎಷ್ಟು?"
    ],
    "my": [
        "တြိဂံတစ်ခုတွင် အနားအလျား 5 စင်တီမီတာနှင့် 12 စင်တီမီတာရှိပြီး ၎င်းတို့ကြားတွင် ထောင့်မှန်တစ်ခုရှိသည်။ တတိယအနား၏ အလျားသည် ဘယ်လောက်လဲ။",
        "ကိုယ်စားလှယ်လောင်း ၅ ဦး (Alice, Bob, Charlie, Dave, Eve) ပါဝင်သော အဖွဲ့မှ ကော်မတီဝင် ၂ ဦးကို ရွေးချယ်ရန် လိုအပ်သည်။ သို့သော် Alice နှင့် Bob သည် ကော်မတီတွင် အတူတူ မရှိနိုင်ပါ။ ကော်မတီကို နည်းလမ်းဘယ်နှစ်မျိုးဖြင့် ဖွဲ့စည်းနိုင်သနည်း။",
        "စက် A သည် တစ်နာရီလျှင် ယူနစ် ၁၀၀ နှုန်းဖြင့် ၈ နာရီကြာ လည်ပတ်သည်။ စက် B သည် ၄ နာရီကြာ လည်ပတ်ပြီး ၁ နာရီ နားကာ၊ ထို့နောက် တစ်နာရီလျှင် ယူနစ် ၁၂၀ နှုန်းဖြင့် ၃ နာရီကြာ လည်ပတ်သည်။ စုစုပေါင်း ထုတ်လုပ်မှု ပမာဏ ဘယ်လောက်လဲ။"
    ],
    "km": [
        "ត្រីកោណមួយមានជ្រុងប្រវែង 5 សង់ទីម៉ែត្រ និង 12 សង់ទីម៉ែត្រ ដោយមានមុំកែងនៅចន្លោះជ្រុងទាំងនោះ។ តើជ្រុងទីបីមានប្រវែងប៉ុន្មាន?",
        "គណៈកម្មាធិការមួយត្រូវជ្រើសរើសសមាជិក 2 នាក់ពីក្រុមបេក្ខជន 5 នាក់ (Alice, Bob, Charlie, Dave, Eve)។ ទោះជាយ៉ាងណាក៏ដោយ Alice និង Bob មិនអាចនៅជាមួយគ្នាក្នុងគណៈកម្មាធិការបានទេ។ តើគណៈកម្មាធិការអាចត្រូវបានបង្កើតឡើងដោយរបៀបណាខ្លះ?",
        "ម៉ាស៊ីន A ដំណើរការរយៈពេល 8 ម៉ោងដោយផលិតបាន 100 ឯកតា/ម៉ោង។ ម៉ាស៊ីន B ដំណើរការរយៈពេល 4 ម៉ោង សម្រាក 1 ម៉ោង បន្ទាប់មកដំណើរការរយៈពេល 3 ម៉ោងក្នុងអត្រា 120 ឯកតា/ម៉ោង។ តើការផលិតសរុបមានចំនួនប៉ុន្មាន?"
    ],
    "am": [
        "አንድ ሶስት ማዕዘን 5 ሴ.ሜ እና 12 ሴ.ሜ ርዝመት ያላቸው ጎኖች አሉት፣ በመካከላቸውም ቀኝ ማዕዘን አለ። የሶስተኛው ጎን ርዝመት ስንት ነው?",
        "አንድ ኮሚቴ ከ 5 እጩዎች (Alice, Bob, Charlie, Dave, Eve) ቡድን 2 አባላትን መምረጥ አለበት። ሆኖም፣ Alice እና Bob በኮሚቴው ውስጥ አብረው መሆን አይችሉም። ኮሚቴው በስንት መንገዶች ሊዋቀር ይችላል?",
        "ማሽን A በሰዓት 100 አሃዶችን እያመረተ ለ8 ሰዓታት ይሰራል። ማሽን B ለ4 ሰዓታት ይሰራል፣ ለ1 ሰዓት ያርፋል፣ ከዚያም በሰዓት 120 አሃዶችን እያመረተ ለ3 ሰዓታት ይሰራል። አጠቃላይ ምርቱ ስንት ነው?"
    ],
    "yo": [
        "Onigun mẹta kan ni awọn ẹgbẹ ti gigun wọn jẹ 5 cm ati 12 cm, pẹlu igun 90° (right angle) laarin wọn. Kini gigun ti ẹgbẹ kẹta?",
        "Igbimọ kan nilo lati mu awọn ọmọ ẹgbẹ 2 lati inu ẹgbẹ awọn oludije 5 (Alice, Bob, Charlie, Dave, Eve). Sibẹsibẹ, Alice ati Bob ko le wa ninu igbimọ papọ. Awọn ọna melo ni a le fi ṣẹda igbimọ naa?",
        "Ẹrọ A n ṣiṣẹ fun wakati 8 ti o n ṣe awọn ẹyọ 100 fun wakati kan. Ẹrọ B n ṣiṣẹ fun wakati 4, sinmi fun wakati 1, lẹhinna ṣiṣẹ fun wakati 3 ni awọn ẹyọ 120 fun wakati kan. Kini apapọ iṣelọpọ?"
    ],
    "si": [
        "ත්‍රිකෝණයක පැතිවල දිග 5 cm සහ 12 cm වන අතර ඒවා අතර ඍජු කෝණයක් ඇත. තුන්වන පැත්තේ දිග කීයද?",
        "කමිටුවක් සඳහා අපේක්ෂකයින් 5 දෙනෙකුගෙන් (Alice, Bob, Charlie, Dave, Eve) සාමාජිකයින් 2 දෙනෙකු තෝරා ගත යුතුය. කෙසේ වෙතත්, Alice සහ Bob එකම කමිටුවක සිටිය නොහැක. කමිටුව ආකාර කියකට පිහිටුවිය හැකිද?",
        "A යන්ත්‍රය පැයට ඒකක 100 බැගින් පැය 8 ක් ක්‍රියාත්මක වේ. B යන්ත්‍රය පැය 4 ක් ක්‍රියාත්මක වී, පැය 1 ක් විවේක ගෙන, පසුව පැයට ඒකක 120 බැගින් පැය 3 ක් ක්‍රියාත්මක වේ. මුළු නිෂ්පාදනය කොපමණද?"
    ],
    "gu": [
        "એક ત્રિકોણની બાજુઓની લંબાઈ 5 સેમી અને 12 સેમી છે, અને તેમની વચ્ચે કાટખૂણો છે. ત્રીજી બાજુની લંબાઈ કેટલી છે?",
        "એક સમિતિને 5 ઉમેદવારો (Alice, Bob, Charlie, Dave, Eve) ના જૂથમાંથી 2 સભ્યો પસંદ કરવાની જરૂર છે. જો કે, Alice અને Bob સમિતિમાં એકસાથે હોઈ શકતા નથી. સમિતિ કેટલી રીતે રચી શકાય?",
        "મશીન A કલાકના 100 યુનિટના દરે 8 કલાક ચાલે છે. મશીન B 4 કલાક ચાલે છે, 1 કલાક આરામ કરે છે, અને પછી કલાકના 120 યુનિટના દરે 3 કલાક ચાલે છે. કુલ ઉત્પાદન કેટલું છે?"
    ],
    "ne": [
        "एउटा त्रिभुजको दुई भुजाको लम्बाइ ५ सेमी र १२ सेमी छ र तिनीहरूको बीचमा समकोण छ। तेस्रो भुजाको लम्बाइ कति हुन्छ?",
        "एउटा समितिले ५ उम्मेदवारहरूको समूह (Alice, Bob, Charlie, Dave, Eve) बाट २ सदस्य छान्नुपर्नेछ। तर, Alice र Bob समितिमा सँगै हुन सक्दैनन्। समिति कति तरिकाले बनाउन सकिन्छ?",
        "मेशिन A ले प्रति घण्टा १०० एकाइ उत्पादन गर्दै ८ घण्टासम्म चल्छ। मेशिन B ४ घण्टा चल्छ, १ घण्टा आराम गर्छ, र त्यसपछि प्रति घण्टा १२० एकाइको दरले ३ घण्टा चल्छ। कुल उत्पादन कति हुन्छ?"
    ],
    "uz": [
        "Uchburchakning tomonlari 5 sm va 12 sm uzunlikda bo'lib, ular orasida to'g'ri burchak mavjud. Uchinchi tomonning uzunligi qancha?",
        "Qo'mita 5 nafar nomzod (Alice, Bob, Charlie, Dave, Eve) orasidan 2 nafar a'zoni tanlab olishi kerak. Biroq, Alice va Bob qo'mitada birga bo'la olmaydi. Qo'mitani necha xil usulda tuzish mumkin?",
        "A mashinasi soatiga 100 birlik ishlab chiqarib, 8 soat ishlaydi. B mashinasi 4 soat ishlaydi, 1 soat dam oladi, so'ngra soatiga 120 birlik tezlikda 3 soat ishlaydi. Jami ishlab chiqarish qancha?"
    ],
    "ky": [
        "Үч бурчтуктун жактарынын узундугу 5 см жана 12 см, алардын ортосунда тик бурч бар. Үчүнчү жактын узундугу канча?",
        "Комитет 5 талапкерден (Alice, Bob, Charlie, Dave, Eve) турган топтон 2 мүчөнү тандап алышы керек. Бирок, Alice менен Bob комитетте чогуу боло алышпайт. Комитетти канча жол менен түзүүгө болот?",
        "А машинасы саатына 100 бирдик өндүрүп, 8 саат иштейт. В машинасы 4 саат иштейт, 1 саат эс алат, андан кийин саатына 120 бирдик ылдамдыкта 3 саат иштейт. Жалпы өндүрүш канча?"
    ],
    "ceb": [
        "Usa ka triyanggulo adunay mga kilid nga may gitas-on nga 5 cm ug 12 cm, nga adunay tuo nga anggulo sa taliwala nila. Unsa ang gitas-on sa ikatulo nga kilid?",
        "Usa ka komite kinahanglan nga mopili og 2 ka miyembro gikan sa grupo sa 5 ka kandidato (Alice, Bob, Charlie, Dave, Eve). Apan, si Alice ug Bob dili mahimong magkuyog sa komite. Pila ka paagi maporma ang komite?",
        "Ang Makina A nagdagan sulod sa 8 ka oras nga naghimo og 100 ka yunit matag oras. Ang Makina B nagdagan sulod sa 4 ka oras, nagpahulay sulod sa 1 ka oras, dayon nagdagan sulod sa 3 ka oras sa 120 ka yunit matag oras. Unsa ang kinatibuk-ang produksyon?"
    ],
    "eu": [
        "Triangelu batek 5 cm eta 12 cm-ko luzera duten aldeak ditu, haien artean angelu zuzena duela. Zein da hirugarren aldearen luzera?",
        "Batzorde batek 2 kide aukeratu behar ditu 5 hautagaiko talde batetik (Alice, Bob, Charlie, Dave, Eve). Hala ere, Alice eta Bob ezin dira elkarrekin egon batzordean. Zenbat modutan osa daiteke batzordea?",
        "A makinak 8 orduz funtzionatzen du orduko 100 unitate ekoizten. B makinak 4 orduz funtzionatzen du, ordu 1ez atseden hartzen du, eta ondoren 3 orduz funtzionatzen du orduko 120 unitateko abiaduran. Zein da ekoizpen osoa?"
    ],
    "gn": [
        "Peteĩ triángulo oreko lado 5 cm ha 12 cm, ha oreko peteĩ ángulo recto imbytépe. Mboýpa ipuku pe tercer lado?",
        "Peteĩ komite oiporavova'erã 2 miyembro 5 candidato apytégui (Alice, Bob, Charlie, Dave, Eve). Upéicharamo jepe, Alice ha Bob ndaikatúi oĩ oñondive pe komitépe. Mboýpa ikatu oñemohenda pe komite?",
        "Máquina A omba'apo 8 hora ha ojapo 100 unidad cada hora. Máquina B omba'apo 4 hora, opytu'u 1 hora, ha upéi omba'apo 3 hora 120 unidad cada hora. Mboýpa pe producción total?"
    ],
    "hy": [
        "Եռանկյունն ունի 5 սմ և 12 սմ երկարությամբ կողմեր, որոնց միջև կա ուղիղ անկյուն: Որքա՞ն է երրորդ կողմի երկարությունը:",
        "Հանձնաժողովը պետք է ընտրի 2 անդամ 5 թեկնածուներից բաղկացած խմբից (Ալիս, Բոբ, Չարլի, Դեյվ, Եվա): Սակայն Ալիսը և Բոբը չեն կարող միասին լինել հանձնաժողովում: Քանի՞ եղանակով կարելի է կազմել հանձնաժողովը:",
        "A մեքենան աշխատում է 8 ժամ՝ ժամում արտադրելով 100 միավոր: B մեքենան աշխատում է 4 ժամ, հանգստանում է 1 ժամ, այնուհետև աշխատում է 3 ժամ՝ ժամում 120 միավոր արագությամբ: Որքա՞ն է ընդհանուր արտադրանքը:"
    ],
    "jv": [
        "Segitiga nduweni sisi kang dawane 5 cm lan 12 cm, kanthi sudut siku-siku ing antarane. Pira dawane sisi sing ketelu?",
        "Sawijining panitia kudu milih 2 anggota saka klompok 5 calon (Alice, Bob, Charlie, Dave, Eve). Nanging, Alice lan Bob ora bisa dadi siji ing panitia. Pira cara panitia bisa dibentuk?",
        "Mesin A mlaku suwene 8 jam ngasilake 100 unit/jam. Mesin B mlaku suwene 4 jam, leren 1 jam, banjur mlaku 3 jam kanthi 120 unit/jam. Pira total produksine?"
    ],
    "ka": [
        "სამკუთხედს აქვს 5 სმ და 12 სმ სიგრძის გვერდები, მათ შორის მართი კუთხით. რა სიგრძისაა მესამე გვერდი?",
        "კომიტეტმა უნდა აირჩიოს 2 წევრი 5 კანდიდატის ჯგუფიდან (ალისა, ბობი, ჩარლი, დეივი, ევა). თუმცა, ალისა და ბობი კომიტეტში ერთად ვერ იქნებიან. რამდენი გზით შეიძლება კომიტეტის შექმნა?",
        "მანქანა A მუშაობს 8 საათის განმავლობაში და აწარმოებს 100 ერთეულს საათში. მანქანა B მუშაობს 4 საათის განმავლობაში, ისვენებს 1 საათი, შემდეგ კი მუშაობს 3 საათი 120 ერთეული საათში სიჩქარით. როგორია მთლიანი წარმოება?"
    ],
    "kk": [
        "Үшбұрыштың қабырғаларының ұзындығы 5 см және 12 см, олардың арасында тік бұрыш бар. Үшінші қабырғаның ұзындығы қандай?",
        "Комитет 5 үміткерден (Алиса, Боб, Чарли, Дэйв, Ева) тұратын топтан 2 мүшені таңдауы керек. Алайда, Алиса мен Боб комитетте бірге бола алмайды. Комитетті неше жолмен құруға болады?",
        "A машинасы сағатына 100 дана шығарып, 8 сағат жұмыс істейді. B машинасы 4 сағат жұмыс істейді, 1 сағат демалады, содан кейін сағатына 120 дана жылдамдықпен 3 сағат жұмыс істейді. Жалпы өндіріс қанша?"
    ],
    "ku": [
        "Sêgoşeyek heye ku dirêjahiya aliyên wê 5 cm û 12 cm ye, û di navbera wan de goşeyek rast heye. Dirêjahiya aliyê sêyemîn çend e?",
        "Komîteyek ji komeke 5 namzedan (Alice, Bob, Charlie, Dave, Eve) 2 endaman hilbijêre. Lêbelê, Alice û Bob nikarin bi hev re di komîteyê de bin. Komîte bi çend awayan dikare were avakirin?",
        "Makîneya A 8 demjimêran dixebite û di saetekê de 100 yekîneyan hilberîne. Makîneya B 4 saetan dixebite, 1 saetê bêhna xwe vedide, piştre 3 saetan bi leza 120 yekîneyan dixebite. Hilberîna giştî çend e?"
    ],
    "lo": [
        "ຮູບສາມແຈມີຂ້າງຍາວ 5 ຊມ ແລະ 12 ຊມ, ໂດຍມີມຸມສາກລະຫວ່າງພວກມັນ. ຂ້າງທີສາມຍາວເທົ່າໃດ?",
        "ຄະນະກໍາມະການຕ້ອງເລືອກສະມາຊິກ 2 ຄົນຈາກກຸ່ມຜູ້ສະຫມັກ 5 ຄົນ (Alice, Bob, Charlie, Dave, Eve). ແນວໃດກໍ່ຕາມ, Alice ແລະ Bob ບໍ່ສາມາດຢູ່ໃນຄະນະກໍາມະການຮ່ວມກັນໄດ້. ຄະນະກໍາມະການສາມາດສ້າງຕັ້ງຂຶ້ນໄດ້ຈັກວິທີ?",
        "ເຄື່ອງຈັກ A ເຮັດວຽກເປັນເວລາ 8 ຊົ່ວໂມງ ຜະລິດໄດ້ 100 ໜ່ວຍ/ຊົ່ວໂມງ. ເຄື່ອງຈັກ B ເຮັດວຽກ 4 ຊົ່ວໂມງ, ພັກຜ່ອນ 1 ຊົ່ວໂມງ, ຈາກນັ້ນເຮັດວຽກ 3 ຊົ່ວໂມງ ດ້ວຍຄວາມໄວ 120 ໜ່ວຍ/ຊົ່ວໂມງ. ການຜະລິດທັງໝົດແມ່ນເທົ່າໃດ?"
    ],
    "mg": [
        "Misy telozoro manana lafiny mirefy 5 sm sy 12 sm, ary misy zoro mahitsy eo anelanelan'izy ireo. Inona ny halavan'ny lafiny fahatelo?",
        "Komity iray no mila mifidy mpikambana 2 avy amin'ny vondrona kandida 5 (Alice, Bob, Charlie, Dave, Eve). Na izany aza, Alice sy Bob dia tsy afaka miara-miasa ao amin'ny komity. Fomba firy no ahafahana mamorona ny komity?",
        "Ny milina A dia miasa 8 ora ary mamokatra singa 100 isan'ora. Ny milina B dia miasa 4 ora, miala sasatra 1 ora, avy eo miasa 3 ora amin'ny tahan'ny 120 singa isan'ora. Inona ny totalin'ny vokatra?"
    ],
    "ml": [
        "ഒരു ത്രികോണത്തിന് 5 സെ.മീ, 12 സെ.മീ നീളമുള്ള വശങ്ങളുണ്ട്, അവയ്ക്കിടയിൽ ഒരു ലംബകോണുമുണ്ട്. മൂന്നാമത്തെ വശത്തിൻ്റെ നീളം എത്രയാണ്?",
        "5 സ്ഥാനാർത്ഥികളുള്ള (Alice, Bob, Charlie, Dave, Eve) ഒരു ഗ്രൂപ്പിൽ നിന്ന് 2 അംഗങ്ങളെ ഒരു കമ്മിറ്റിയിലേക്ക് തിരഞ്ഞെടുക്കേണ്ടതുണ്ട്. എന്നിരുന്നാലും, ആലീസിനും ബോബിനും കമ്മിറ്റിയിൽ ഒന്നിച്ചുണ്ടാകാൻ കഴിയില്ല. എത്ര രീതിയിൽ കമ്മിറ്റി രൂപീകരിക്കാം?",
        "മെഷീൻ എ മണിക്കൂറിൽ 100 യൂണിറ്റ് വീതം 8 മണിക്കൂർ പ്രവർത്തിക്കുന്നു. മെഷീൻ ബി 4 മണിക്കൂർ പ്രവർത്തിക്കുന്നു, 1 മണിക്കൂർ വിശ്രമിക്കുന്നു, പിന്നീട് 3 മണിക്കൂർ 120 യൂണിറ്റ്/മണിക്കൂർ എന്ന നിരക്കിൽ പ്രവർത്തിക്കുന്നു. ആകെ ഉൽപ്പാദനം എത്രയാണ്?"
    ],
    "mn": [
        "Гурвалжин нь 5 см ба 12 см урттай талуудтай бөгөөд тэдгээрийн хооронд тэгш өнцөг үүснэ. Гурав дахь талын урт хэд вэ?",
        "Хороо 5 нэр дэвшигчийн (Алис, Боб, Чарли, Дэйв, Ева) бүлгээс 2 гишүүнийг сонгох шаардлагатай. Гэсэн хэдий ч Алис, Боб хоёр хороонд хамт байх боломжгүй. Хороог хэдэн аргаар байгуулж болох вэ?",
        "А машин цагт 100 нэгж бүтээгдэхүүн үйлдвэрлэж 8 цаг ажилладаг. В машин 4 цаг ажиллаж, 1 цаг амарч, дараа нь цагт 120 нэгж хурдаар 3 цаг ажилладаг. Нийт үйлдвэрлэл хэд вэ?"
    ],
    "mr": [
        "एका त्रिकोणाच्या बाजूंची लांबी 5 सेमी आणि 12 सेमी आहे, आणि त्यांच्यामध्ये काटकोन आहे. तिसऱ्या बाजूची लांबी किती आहे?",
        "एका समितीला 5 उमेदवारांच्या (Alice, Bob, Charlie, Dave, Eve) गटातून 2 सदस्य निवडायचे आहेत. पण, Alice आणि Bob समितीत एकत्र असू शकत नाहीत. समिती किती प्रकारे तयार केली जाऊ शकते?",
        "मशीन A 8 तास चालते आणि प्रति तास 100 युनिट्स उत्पादन करते. मशीन B 4 तास चालते, 1 तास विश्रांती घेते, आणि नंतर 3 तास 120 युनिट्स प्रति तास या दराने चालते. एकूण उत्पादन किती आहे?"
    ],
    "mt": [
        "Trijanglu għandu naħat ta' tul 5 ċm u 12-il ċm, b'angolu rett bejniethom. X'inhu t-tul tat-tielet naħa?",
        "Kumitat irid jagħżel 2 membri minn grupp ta' 5 kandidati (Alice, Bob, Charlie, Dave, Eve). Madankollu, Alice u Bob ma jistgħux ikunu fil-kumitat flimkien. B'kemm-il mod jista' jiġi ffurmat il-kumitat?",
        "Magna A taħdem għal 8 sigħat u tipproduċi 100 unità fis-siegħa. Magna B taħdem għal 4 sigħat, tistrieħ għal siegħa, imbagħad taħdem għal 3 sigħat b'rata ta' 120 unità fis-siegħa. X'inhi l-produzzjoni totali?"
    ],
    "or": [
        "ଏକ ତ୍ରିଭୁଜର ବାହୁଗୁଡ଼ିକର ଦୈର୍ଘ୍ୟ 5 ସେମି ଏବଂ 12 ସେମି ଅଟେ, ଏବଂ ସେମାନଙ୍କ ମଧ୍ୟରେ ଏକ ସମକୋଣ ଅଛି | ତୃତୀୟ ବାହୁର ଦୈର୍ଘ୍ୟ କେତେ?",
        "ଏକ କମିଟିକୁ 5 ଜଣ ପ୍ରାର୍ଥୀ (Alice, Bob, Charlie, Dave, Eve) ଙ୍କ ଗୋଷ୍ଠୀରୁ 2 ଜଣ ସଦସ୍ୟ ଚୟନ କରିବାକୁ ପଡିବ | ତଥାପି, ଆଲିସ୍ ଏବଂ ବବ୍ କମିଟିରେ ଏକାଠି ରହିପାରିବେ ନାହିଁ | କମିଟି କେତେ ଉପାୟରେ ଗଠନ କରାଯାଇପାରିବ?",
        "ମେସିନ୍ A 8 ଘଣ୍ଟା ଚାଲିଥାଏ ଏବଂ ଘଣ୍ଟା ପ୍ରତି 100 ୟୁନିଟ୍ ଉତ୍ପାଦନ କରେ | ମେସିନ୍ B 4 ଘଣ୍ଟା ଚାଲିଥାଏ, 1 ଘଣ୍ଟା ବିଶ୍ରାମ ନିଏ, ତାପରେ ଘଣ୍ଟା ପ୍ରତି 120 ୟୁନିଟ୍ ହାରରେ 3 ଘଣ୍ଟା ଚାଲିଥାଏ | ମୋଟ ଉତ୍ପାଦନ କେତେ?"
    ],
    "pa": [
        "ਇੱਕ ਤਿਕੋਣ ਦੀਆਂ ਭੁਜਾਵਾਂ 5 ਸੈਂਟੀਮੀਟਰ ਅਤੇ 12 ਸੈਂਟੀਮੀਟਰ ਲੰਬੀਆਂ ਹਨ, ਜਿਨ੍ਹਾਂ ਦੇ ਵਿਚਕਾਰ ਇੱਕ ਸਮਕੋਣ ਹੈ। ਤੀਜੀ ਭੁਜਾ ਦੀ ਲੰਬਾਈ ਕਿੰਨੀ ਹੈ?",
        "ਇੱਕ ਕਮੇਟੀ ਨੂੰ 5 ਉਮੀਦਵਾਰਾਂ (Alice, Bob, Charlie, Dave, Eve) ਦੇ ਸਮੂਹ ਵਿੱਚੋਂ 2 ਮੈਂਬਰ ਚੁਣਨ ਦੀ ਲੋੜ ਹੈ। ਹਾਲਾਂਕਿ, ਐਲਿਸ ਅਤੇ ਬੌਬ ਕਮੇਟੀ ਵਿੱਚ ਇਕੱਠੇ ਨਹੀਂ ਹੋ ਸਕਦੇ। ਕਮੇਟੀ ਕਿੰਨੇ ਤਰੀਕਿਆਂ ਨਾਲ ਬਣਾਈ ਜਾ ਸਕਦੀ ਹੈ?",
        "ਮਸ਼ੀਨ A 8 ਘੰਟੇ ਚੱਲਦੀ ਹੈ ਅਤੇ ਪ੍ਰਤੀ ਘੰਟਾ 100 ਯੂਨਿਟ ਪੈਦਾ ਕਰਦੀ ਹੈ। ਮਸ਼ੀਨ B 4 ਘੰਟੇ ਚੱਲਦੀ ਹੈ, 1 ਘੰਟਾ ਆਰਾਮ ਕਰਦੀ ਹੈ, ਫਿਰ 120 ਯੂਨਿਟ ਪ੍ਰਤੀ ਘੰਟਾ ਦੀ ਦਰ ਨਾਲ 3 ਘੰਟੇ ਚੱਲਦੀ ਹੈ। ਕੁੱਲ ਉਤਪਾਦਨ ਕਿੰਨਾ ਹੈ?"
    ],
    "ps": [
        "یو مثلث 5 سانتي متره او 12 سانتي متره اوږدوالی لري، او د دوی ترمنځ یوه سمه زاویه (90 درجې) ده. د دریمې غاړې اوږدوالی څومره دی؟",
        "یوه کمیټه باید د 5 کاندیدانو (Alice, Bob, Charlie, Dave, Eve) له یوې ډلې څخه 2 غړي وټاکي. په هرصورت، ایلیس او باب نشي کولی په کمیټه کې یوځای وي. کمیټه په څو لارو جوړیدلی شي؟",
        "ماشین A د 8 ساعتونو لپاره چالانیږي او په ساعت کې 100 واحدونه تولیدوي. ماشین B د 4 ساعتونو لپاره چالانیږي، 1 ساعت آرام کوي، بیا د 3 ساعتونو لپاره په ساعت کې د 120 واحدونو په سرعت سره چالانیږي. ټول تولید څومره دی؟"
    ],
    "qu": [
        "Huk kimsak'uchu kan 5 cm, 12 cm suni waqtayuq, chawpinkupi paqta k'uchuyuq. Maynatan chay kimsa kaq waqta tupun?",
        "Huk huñunakuy 2 runata akllanan tiyan 5 mañakuqkunamanta (Alice, Bob, Charlie, Dave, Eve). Ichaqa, Alice, Bob ima mana kuska kayta atinkuchu chay huñunakuypi. Hayk'a ñankunapitaq chay huñunakuy ruwakunman?",
        "Maquina A llamk'an 8 horasta, horapi 100 unit-ta ruraspa. Maquina B llamk'an 4 horasta, samarin 1 horata, chaymanta llamk'allantaq 3 horasta 120 unit/horapi. Hayk'ataq llapan rurusqan?"
    ],
    "sd": [
        "هڪ ٽڪنڊي جا پاسا 5 سينٽي ۽ 12 سينٽي ميٽر ڊگھا آهن، جن جي وچ ۾ ڪائمه ڪنڊ (90 degrees) آهي. ٽئين پاسي جي ڊيگهه ڪيتري آهي؟",
        "هڪ ڪميٽي کي 5 اميدوارن (Alice, Bob, Charlie, Dave, Eve) جي گروپ مان 2 ميمبر چونڊڻا آهن. بهرحال، ايلس ۽ بوب ڪميٽي تي گڏ نه ٿا ٿي سگهن. ڪميٽي ڪيترا طريقن سان ٺهي سگهي ٿي؟",
        "مشين A 8 ڪلاڪ هلندي آهي ۽ في ڪلاڪ 100 يونٽ پيدا ڪندي آهي. مشين B 4 ڪلاڪ هلندي آهي، 1 ڪلاڪ آرام ڪندي آهي، پوءِ 3 ڪلاڪ 120 يونٽ في ڪلاڪ جي رفتار سان هلندي آهي. ڪل پيداوار ڇا آهي؟"
    ],
    "so": [
        "Saddex-xagal wuxuu leeyahay dhinacyo dhererkoodu egyahay 5 cm iyo 12 cm, oo xagal quman u dhaxayso. Waa imisa dhererka dhinaca saddexaad?",
        "Guddi waa inay 2 xubnood ka doortaan koox ka kooban 5 musharax (Alice, Bob, Charlie, Dave, Eve). Si kastaba ha ahaatee, Alice iyo Bob iskuma joogi karaan guddiga. Immisa siyaabood ayaa guddiga loo samayn karaa?",
        "Mashiinka A wuxuu shaqeeyaa 8 saacadood isagoo soo saaraya 100 unug/saacaddii. Mashiinka B wuxuu shaqeeyaa 4 saacadood, wuu nasiyayaa 1 saac, ka dibna wuxuu shaqeeyaa 3 saacadood iyadoo xawaarihiisu yahay 120 unug/saacaddii. Waa imisa wadarta wax soo saarku?"
    ],
    "su": [
        "Hiji segitiga boga sisi panjangna 5 cm jeung 12 cm, kalawan sudut siku-siku di antara maranéhanana. Sabaraha panjang sisi katilu?",
        "Panitia kudu milih 2 anggota tina grup 5 calon (Alice, Bob, Charlie, Dave, Eve). Tapi, Alice jeung Bob teu bisa babarengan dina panitia. Sabaraha cara panitia bisa dijieun?",
        "Mesin A jalan salila 8 jam ngahasilkeun 100 unit/jam. Mesin B jalan salila 4 jam, istirahat 1 jam, tuluy jalan 3 jam kalawan 120 unit/jam. Sabaraha total produksina?"
    ],
    "tg": [
        "Сегӯша дорои паҳлӯҳои дарозиашон 5 см ва 12 см буда, дар байни онҳо кунҷи рост (90 дараҷа) ҷойгир аст. Дарозии паҳлӯи сеюм чанд аст?",
        "Кумита бояд аз гурӯҳи 5 номзад (Alice, Bob, Charlie, Dave, Eve) 2 узвро интихоб кунад. Аммо, Алис ва Боб наметавонанд дар кумита якҷоя бошанд. Кумитаро бо чанд роҳ метавон ташкил кард?",
        "Мошини А 8 соат кор мекунад ва дар як соат 100 воҳид истеҳсол мекунад. Мошини В 4 соат кор мекунад, 1 соат истироҳат мекунад ва сипас 3 соат бо суръати 120 воҳид дар як соат кор мекунад. Истеҳсоли умумӣ чанд аст?"
    ],
    "ug": [
        "بىر ئۈچبۇلۇڭنىڭ تەرەپلىرى 5 سانتىمېتىر ۋە 12 سانتىمېتىر بولۇپ ، ئۇلارنىڭ ئارىسىدا تىك بۇلۇڭ (right angle) بار. ئۈچىنچى تەرەپنىڭ ئۇزۇنلۇقى قانچىلىك؟",
        "بىر كومىتېت 5 نامزات (Alice, Bob, Charlie, Dave, Eve) گۇرۇپپىسىدىن 2 ئەزانى تاللىشى كېرەك. لېكىن ، Alice بىلەن Bob كومىتېتتا بىللە بولالمايدۇ. كومىتېتنى قانچە خىل ئۇسۇلدا قۇرغىلى بولىدۇ؟",
        "ماشىنا A 8 سائەت ئىشلەيدۇ ، سائىتىگە 100 بىرلىك ئىشلەپچىقىرىدۇ. ماشىنا B 4 سائەت ئىشلەيدۇ ، 1 سائەت دەم ئالىدۇ ، ئاندىن 3 سائەت 120 بىرلىك/سائەت سۈرئەتتە ئىشلەيدۇ. ئومۇمىي مەھسۇلات قانچىلىك؟"
    ],
}

SKELETON_FEWSHOTS = {}
for lang_code, questions in SKELETON_QUESTIONS.items():
    shots = []
    for q, a in zip(questions, SKELETON_ANSWERS):
        shots.append({"q": q, "a": a})
    SKELETON_FEWSHOTS[lang_code] = shots

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
    messages = []
    
    if few_shots:
        for shot in few_shots:
            messages.append({"role": "user", "content": shot["question"]})
            messages.append({"role": "assistant", "content": shot["answer"]})
    
    # Determine language for reasoning
    reason_lang = "en" if force_english else lang
    
    lang_triggers = {
        # 인도 언어들
        "ta": "சரி,\n",           # Tamil (Sari - Okay)
        "kn": "ಸರಿ,\n",           # Kannada (Sari - Okay)
        "si": "හොඳයි,\n",         # Sinhala (Hondai - Good)
        "gu": "સારું,\n",          # Gujarati (Saru - Good)
        "ne": "हुन्छ,\n",          # Nepali (Huncha - Okay/Will do)
        "ml": "ശരി,\n",           # Malayalam (Sari - Okay)
        "mr": "ठीक,\n",           # Marathi (Thik - Okay)
        "or": "ଠିକ୍,\n",           # Odia (Thik - Okay)
        "pa": "ਠੀਕ,\n",           # Punjabi (Thik - Okay)
        
        # 동남아시아
        "my": "ကောင်းပြီ,\n",      # Burmese (Kaung byi - Good)
        "km": "ល្អ,\n",            # Khmer (L'or - Good)
        "lo": "ໂດຍ,\n",            # Lao (Doy - Polite Yes/Ack) *수정됨*
        "jv": "Inggih,\n",         # Javanese (Inggih - Polite Yes) *수정됨*
        "su": "Leres,\n",         # Sundanese (Leres - Right/Polite)
        "ceb": "Sige,\n",         # Cebuano (Sige - Go ahead/Okay)
        
        # 중앙아시아 / 튀르크어
        "uz": "Yaxshi,\n",        # Uzbek (Good)
        "ky": "Жакшы,\n",         # Kyrgyz (Good)
        "kk": "Жақсы,\n",         # Kazakh (Good)
        "tg": "Хуб,\n",           # Tajik (Good)
        "ug": "ياخشى,\n",          # Uyghur (Yaxshi - Good)
        
        # 캅카스 / 아르메니아
        "ka": "კარგი,\n",          # Georgian (Kargi - Good)
        "hy": "Լավ,\n",            # Armenian (Lav - Good) *수정됨*
        
        # 아프리카
        "am": "እሺ,\n",            # Amharic (Eshi - Okay)
        "yo": "O da,\n",          # Yoruba (It is good)
        "so": "Hagaag,\n",        # Somali (Fine/Okay)
        "mg": "Eny,\n",           # Malagasy (Yes/Okay)
        
        # 유럽
        "eu": "Ados,\n",          # Basque (Agreed)
        "mt": "Tajjeb,\n",        # Maltese (Good)
        
        # 중동
        "ku": "Baş e,\n",         # Kurdish (It is good)
        "ps": "ښه,\n",            # Pashto (Good)
        "sd": "ٺيڪ,\n",           # Sindhi (Okay)
        
        # 아메리카 원주민
        "gn": "Oĩma,\n",          # Guarani (Ready/Okay)
        "qu": "Allin,\n",         # Quechua (Good)
        
        # 기타
        "mn": "За,\n",            # Mongolian (Okay/Well)
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
    
    lang_triggers = {
        # 인도 언어들
        "ta": "சரி,\n",           # Tamil (Sari - Okay)
        "kn": "ಸರಿ,\n",           # Kannada (Sari - Okay)
        "si": "හොඳයි,\n",         # Sinhala (Hondai - Good)
        "gu": "સારું,\n",          # Gujarati (Saru - Good)
        "ne": "हुन्छ,\n",          # Nepali (Huncha - Okay/Will do)
        "ml": "ശരി,\n",           # Malayalam (Sari - Okay)
        "mr": "ठीक,\n",           # Marathi (Thik - Okay)
        "or": "ଠିକ୍,\n",           # Odia (Thik - Okay)
        "pa": "ਠੀਕ,\n",           # Punjabi (Thik - Okay)
        
        # 동남아시아
        "my": "ကောင်းပြီ,\n",      # Burmese (Kaung byi - Good)
        "km": "ល្អ,\n",            # Khmer (L'or - Good)
        "lo": "ໂດຍ,\n",            # Lao (Doy - Polite Yes/Ack) *수정됨*
        "jv": "Inggih,\n",         # Javanese (Inggih - Polite Yes) *수정됨*
        "su": "Leres,\n",         # Sundanese (Leres - Right/Polite)
        "ceb": "Sige,\n",         # Cebuano (Sige - Go ahead/Okay)
        
        # 중앙아시아 / 튀르크어
        "uz": "Yaxshi,\n",        # Uzbek (Good)
        "ky": "Жакшы,\n",         # Kyrgyz (Good)
        "kk": "Жақсы,\n",         # Kazakh (Good)
        "tg": "Хуб,\n",           # Tajik (Good)
        "ug": "ياخشى,\n",          # Uyghur (Yaxshi - Good)
        
        # 캅카스 / 아르메니아
        "ka": "კარგი,\n",          # Georgian (Kargi - Good)
        "hy": "Լավ,\n",            # Armenian (Lav - Good) *수정됨*
        
        # 아프리카
        "am": "እሺ,\n",            # Amharic (Eshi - Okay)
        "yo": "O da,\n",          # Yoruba (It is good)
        "so": "Hagaag,\n",        # Somali (Fine/Okay)
        "mg": "Eny,\n",           # Malagasy (Yes/Okay)
        
        # 유럽
        "eu": "Ados,\n",          # Basque (Agreed)
        "mt": "Tajjeb,\n",        # Maltese (Good)
        
        # 중동
        "ku": "Baş e,\n",         # Kurdish (It is good)
        "ps": "ښه,\n",            # Pashto (Good)
        "sd": "ٺيڪ,\n",           # Sindhi (Okay)
        
        # 아메리카 원주민
        "gn": "Oĩma,\n",          # Guarani (Ready/Okay)
        "qu": "Allin,\n",         # Quechua (Good)
        
        # 기타
        "mn": "За,\n",            # Mongolian (Okay/Well)
    }

    
    if reasoning_model:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        trigger = lang_triggers.get(actual_lang, "")
        if trigger:
            text += f"<think>\n{trigger}"
    else:
        lang_name = get_lang_name(actual_lang)
        if actual_lang != "en":
            cot_hint = f"Let's think step by step. Respond in {lang_name}."
            forcing = lang_triggers.get(actual_lang, "")
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
        all_langs = ["ta", "kn", "my", "km", "am", "yo", "si", "gu", "ne", "uz", "ky", "ceb", "eu", "gn", "hy", "jv", "ka", "kk", "ku", "lo", "mg", "ml", "mn", "mr", "mt", "or", "pa", "ps", "qu", "sd", "so", "su", "tg", "ug"]
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
                    
                    # [Optional Turn 0] Translate question to English
                    if args.translate_q:
                        trans_prompts = []
                        trans_indices = []
                        for i, t in enumerate(batch):
                            if t["lang"] != "en":
                                trans_prompts.append(build_translation_prompt(tok, t["question"], t["lang"], "en"))
                                trans_indices.append(i)
                        
                        if trans_prompts:
                            trans_outputs = llm.generate(trans_prompts, trans_sp, use_tqdm=False)
                            for idx, out in zip(trans_indices, trans_outputs):
                                if out.outputs:
                                    translated_q = out.outputs[0].text.strip()
                                    translated_questions[idx] = translated_q
                                    questions_to_use[idx] = translated_q
                    
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
                            # "input_prompt": sol_prompts[i],
                            "skeleton_prompt": sk_prompts[i],
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
                    has_multilang = any([row.get(f"question_{l}") for l in ["ta", "kn", "my", "km", "am", "yo", "si", "gu", "ne", "uz", "ky", "ceb", "eu", "gn", "hy", "jv", "ka", "kk", "ku", "lo", "mg", "ml", "mn", "mr", "mt", "or", "pa", "ps", "qu", "sd", "so", "su", "tg", "ug"]])
                    
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
        gen_sp = SamplingParams(temperature=eff_temp, max_tokens=args.max_tokens, 
                                top_k=args.top_k, top_p=eff_top_p, n=args.rollout)
        sk_sp = SamplingParams(temperature=eff_temp, max_tokens=args.max_tokens, 
                               top_k=args.top_k, top_p=eff_top_p)
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
