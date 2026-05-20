
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
Always respond in {target_lang}, regardless of the language used in the user query.

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
        "col_map": {"question": "question", "answer": "answer", "id": "id"},
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
#                              ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined Evaluation for Low-Resource Languages with Skeleton Language Control"
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
    parser.add_argument("--langs", type=str, default="ta,kn,my,km,am,yo,si,gu,ne,uz,ky")
    parser.add_argument("--difficulties", type=str, default="top,high,medium,low")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Skeleton language control
    parser.add_argument("--skeleton_lang", type=str, default="en",
                        help="Language for skeleton generation (en, es, ko, zh, ru, th)")
    
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


def get_skeleton_system_prompt(skeleton_lang: str) -> str:
    """Get skeleton system prompt with target language inserted."""
    target_lang = get_lang_name(skeleton_lang)
    return SYSTEM_PROMPT_SKELETON.format(target_lang=target_lang)


# ═══════════════════════════════════════════════════════════════════════════════
#                        SKELETON FEW-SHOTS (Questions in low-resource langs)
# ═══════════════════════════════════════════════════════════════════════════════

SKELETON_FEWSHOTS = {
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

# Skeleton answers in different languages
SKELETON_ANSWER_FEWSHOTS = {
    "en": [
        "**Problem Structure**\n- Geometric figure: right triangle with three sides\n- Known quantities: two perpendicular sides (5 cm, 12 cm), right angle constraint\n- Unknown: third side (hypotenuse) connecting the two legs\n- Relationship: side lengths satisfy distance relation enforced by right angle property\n\n**Key Concepts / Tools**\n- Pythagorean theorem\n- Right triangle geometry",
        "**Problem Structure**\n- Discrete selection problem: choosing k=2 elements from n=5 candidates\n- Known: total candidates (5), committee size (2), pairwise exclusion constraint\n- Unknown: number of valid selection configurations\n- Constraint: pairwise exclusion (Alice and Bob cannot co-occur)\n\n**Key Concepts / Tools**\n- Combinatorial counting (combinations)\n- Constraint satisfaction\n- Complementary counting",
        "**Problem Structure**\n- Two independent production processes with different temporal patterns\n- Variables: production rates, time durations, total output\n- Machine A: single continuous active interval (8 hours @ 100 units/hour)\n- Machine B: two disjoint active intervals (4h + 3h @ 120 units/hour)\n\n**Key Concepts / Tools**\n- Rate-time-quantity relationships\n- Piecewise continuous processes\n- Linear aggregation"
    ],
    "ko": [
        "**문제 구조**\n- 기하학적 도형: 세 변을 가진 직각삼각형\n- 알려진 양: 두 수직 변 (5 cm, 12 cm), 직각 조건\n- 미지수: 두 다리를 연결하는 세 번째 변 (빗변)\n- 관계: 변의 길이는 직각 속성에 의해 강제되는 거리 관계를 만족\n\n**핵심 개념 / 도구**\n- 피타고라스 정리\n- 직각삼각형 기하학",
        "**문제 구조**\n- 이산 선택 문제: n=5명의 후보에서 k=2명의 요소 선택\n- 알려진 것: 총 후보 수 (5), 위원회 크기 (2), 쌍별 배제 제약\n- 미지수: 유효한 선택 구성의 수\n- 제약: 쌍별 배제 (Alice와 Bob은 함께 있을 수 없음)\n\n**핵심 개념 / 도구**\n- 조합 계산 (조합)\n- 제약 충족\n- 여집합 계산",
        "**문제 구조**\n- 서로 다른 시간 패턴을 가진 두 개의 독립적인 생산 프로세스\n- 변수: 생산율, 시간 지속, 총 출력\n- 기계 A: 단일 연속 활성 구간 (8시간 @ 100 단위/시간)\n- 기계 B: 두 개의 분리된 활성 구간 (4시간 + 3시간 @ 120 단위/시간)\n\n**핵심 개념 / 도구**\n- 속도-시간-수량 관계\n- 구간별 연속 프로세스\n- 선형 집계"
    ],
    "es": [
        "**Estructura del Problema**\n- Figura geométrica: triángulo rectángulo con tres lados\n- Cantidades conocidas: dos lados perpendiculares (5 cm, 12 cm), restricción de ángulo recto\n- Incógnita: tercer lado (hipotenusa) que conecta los dos catetos\n- Relación: las longitudes de los lados satisfacen la relación de distancia impuesta por la propiedad del ángulo recto\n\n**Conceptos Clave / Herramientas**\n- Teorema de Pitágoras\n- Geometría del triángulo rectángulo",
        "**Estructura del Problema**\n- Problema de selección discreta: elegir k=2 elementos de n=5 candidatos\n- Conocido: total de candidatos (5), tamaño del comité (2), restricción de exclusión por pares\n- Incógnita: número de configuraciones de selección válidas\n- Restricción: exclusión por pares (Alice y Bob no pueden coincidir)\n\n**Conceptos Clave / Herramientas**\n- Conteo combinatorio (combinaciones)\n- Satisfacción de restricciones\n- Conteo complementario",
        "**Estructura del Problema**\n- Dos procesos de producción independientes con diferentes patrones temporales\n- Variables: tasas de producción, duraciones de tiempo, producción total\n- Máquina A: intervalo activo continuo único (8 horas @ 100 unidades/hora)\n- Máquina B: dos intervalos activos disjuntos (4h + 3h @ 120 unidades/hora)\n\n**Conceptos Clave / Herramientas**\n- Relaciones tasa-tiempo-cantidad\n- Procesos continuos por tramos\n- Agregación lineal"
    ],
    "zh": [
        "**问题结构**\n- 几何图形：具有三条边的直角三角形\n- 已知量：两条垂直边（5厘米，12厘米），直角约束\n- 未知量：连接两条直角边的第三边（斜边）\n- 关系：边长满足由直角属性强制执行的距离关系\n\n**关键概念/工具**\n- 勾股定理\n- 直角三角形几何",
        "**问题结构**\n- 离散选择问题：从n=5个候选人中选择k=2个元素\n- 已知：候选人总数（5），委员会规模（2），成对排斥约束\n- 未知量：有效选择配置的数量\n- 约束：成对排斥（Alice和Bob不能同时出现）\n\n**关键概念/工具**\n- 组合计数（组合）\n- 约束满足\n- 补集计数",
        "**问题结构**\n- 两个具有不同时间模式的独立生产过程\n- 变量：生产率，时间持续，总产量\n- 机器A：单个连续活动区间（8小时 @ 100单位/小时）\n- 机器B：两个不相交的活动区间（4小时 + 3小时 @ 120单位/小时）\n\n**关键概念/工具**\n- 速率-时间-数量关系\n- 分段连续过程\n- 线性聚合"
    ],
    "th": [
        "**โครงสร้างปัญหา**\n- รูปทรงเรขาคณิต: สามเหลี่ยมมุมฉากที่มีด้านสามด้าน\n- ปริมาณที่ทราบ: ด้านตั้งฉากสองด้าน (5 ซม., 12 ซม.), ข้อจำกัดมุมฉาก\n- ไม่ทราบ: ด้านที่สาม (ด้านตรงข้ามมุมฉาก) ที่เชื่อมต่อขาทั้งสอง\n- ความสัมพันธ์: ความยาวด้านเป็นไปตามความสัมพันธ์ระยะทางที่บังคับโดยคุณสมบัติมุมฉาก\n\n**แนวคิดหลัก / เครื่องมือ**\n- ทฤษฎีบทพีทาโกรัส\n- เรขาคณิตสามเหลี่ยมมุมฉาก",
        "**โครงสร้างปัญหา**\n- ปัญหาการเลือกแบบไม่ต่อเนื่อง: เลือก k=2 องค์ประกอบจาก n=5 ผู้สมัคร\n- ทราบ: ผู้สมัครทั้งหมด (5), ขนาดคณะกรรมการ (2), ข้อจำกัดการยกเว้นคู่\n- ไม่ทราบ: จำนวนการกำหนดค่าการเลือกที่ถูกต้อง\n- ข้อจำกัด: การยกเว้นคู่\n\n**แนวคิดหลัก / เครื่องมือ**\n- การนับแบบผสม (การรวมกัน)\n- การตอบสนองข้อจำกัด\n- การนับเสริม",
        "**โครงสร้างปัญหา**\n- กระบวนการผลิตอิสระสองกระบวนการที่มีรูปแบบเวลาต่างกัน\n- ตัวแปร: อัตราการผลิต, ระยะเวลา, ผลผลิตรวม\n- เครื่องจักร A: ช่วงเวลาทำงานต่อเนื่องเดียว (8 ชั่วโมง @ 100 หน่วย/ชั่วโมง)\n- เครื่องจักร B: ช่วงเวลาทำงานแยกกันสองช่วง (4 ชม. + 3 ชม. @ 120 หน่วย/ชั่วโมง)\n\n**แนวคิดหลัก / เครื่องมือ**\n- ความสัมพันธ์อัตรา-เวลา-ปริมาณ\n- กระบวนการต่อเนื่องแบบชิ้นส่วน\n- การรวมแบบเชิงเส้น"
    ],
    "ru": [
        "**Структура задачи**\n- Геометрическая фигура: прямоугольный треугольник с тремя сторонами\n- Известные величины: две перпендикулярные стороны (5 см, 12 см), ограничение прямого угла\n- Неизвестное: третья сторона (гипотенуза), соединяющая два катета\n- Отношение: длины сторон удовлетворяют соотношению расстояний, обусловленному свойством прямого угла\n\n**Ключевые концепции / Инструменты**\n- Теорема Пифагора\n- Геометрия прямоугольного треугольника",
        "**Структура задачи**\n- Дискретная задача выбора: выбор k=2 элементов из n=5 кандидатов\n- Известно: общее количество кандидатов (5), размер комитета (2), ограничение попарного исключения\n- Неизвестное: количество допустимых конфигураций выбора\n- Ограничение: попарное исключение (Алиса и Боб не могут быть вместе)\n\n**Ключевые концепции / Инструменты**\n- Комбинаторный подсчёт (комбинации)\n- Удовлетворение ограничений\n- Дополнительный подсчёт",
        "**Структура задачи**\n- Два независимых производственных процесса с разными временными паттернами\n- Переменные: производственные мощности, временные интервалы, общий выпуск\n- Машина A: единый непрерывный активный интервал (8 часов @ 100 единиц/час)\n- Машина B: два непересекающихся активных интервала (4ч + 3ч @ 120 единиц/час)\n\n**Ключевые концепции / Инструменты**\n- Соотношения скорость-время-количество\n- Кусочно-непрерывные процессы\n- Линейная агрегация"
    ],
}


def build_skeleton_prompt(tok, lang: str, question: str, skeleton_lang: str = "en") -> str:
    """Build skeleton prompt with specified skeleton language for output."""
    fewshots = SKELETON_FEWSHOTS.get(lang, SKELETON_FEWSHOTS["en"])
    system_prompt = get_skeleton_system_prompt(skeleton_lang)
    skeleton_answers = SKELETON_ANSWER_FEWSHOTS.get(skeleton_lang, SKELETON_ANSWER_FEWSHOTS["en"])
    
    messages = [{"role": "system", "content": system_prompt}]
    for i, shot_q in enumerate(fewshots):
        messages.append({"role": "user", "content": shot_q})
        if i < len(skeleton_answers):
            messages.append({"role": "assistant", "content": skeleton_answers[i]})
    
    lang_name = get_lang_name(skeleton_lang)
    messages.append({"role": "user", "content": f"Question: {question}\n\nYou must respond in {lang_name}."})
    
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_solver_prompt(tok, question: str, skeleton: str, lang: str, 
                        few_shots: list = None, force_english: bool = False) -> str:
    """Build solver prompt for skeleton-based solving."""
    messages = []
    
    if few_shots:
        for shot in few_shots:
            messages.append({"role": "user", "content": shot["question"]})
            messages.append({"role": "assistant", "content": shot["answer"]})
    
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
#                           MAIN EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_skeleton_multiturn(args, llm, tok, config, gen_sp, sk_sp, load_targets, langs):
    """Skeleton-based Multi-turn evaluation with skeleton language control."""
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
                
                print(f"🚀 Processing {len(task_queue)} tasks (Skeleton Lang: {args.skeleton_lang})...")
                
                for batch in tqdm(chunks(task_queue, args.batch), total=(len(task_queue) // args.batch + 1)):
                    questions_to_use = [t["question"] for t in batch]
                    
                    # [Turn 1] Skeleton Generation with specified skeleton_lang
                    sk_prompts = [build_skeleton_prompt(tok, t["lang"], q, args.skeleton_lang) 
                                  for t, q in zip(batch, questions_to_use)]
                    
                    if args.lora:
                        sk_outputs = llm.generate(sk_prompts, sk_sp, use_tqdm=False, 
                                                  lora_request=LoRARequest("adapter", 1, lora_adapter))
                    else:
                        sk_outputs = llm.generate(sk_prompts, sk_sp, use_tqdm=False)
                    
                    skeletons_list = [[out.outputs[0].text.strip()] if out.outputs else [""] for out in sk_outputs]
                    
                    # [Turn 2] Solver Generation
                    sol_prompts = []
                    sol_prompt_mapping = []
                    
                    for batch_idx, (t, q, sks) in enumerate(zip(batch, questions_to_use, skeletons_list)):
                        for sk_idx, sk in enumerate(sks):
                            if not sk:
                                continue
                            force_en = args.translate_cot and t["lang"] != "en"
                            sol_prompts.append(build_solver_prompt(tok, q, sk, t["lang"], 
                                                                   force_english=force_en))
                            sol_prompt_mapping.append((batch_idx, sk_idx))
                    
                    sol_texts = [[] for _ in batch]
                    skeletons = [[] for _ in batch]
                    
                    if sol_prompts:
                        sol_outputs = llm.generate(sol_prompts, gen_sp, use_tqdm=False)
                        for (batch_idx, sk_idx), out in zip(sol_prompt_mapping, sol_outputs):
                            if out.outputs:
                                sol_texts[batch_idx].extend([o.text.strip() for o in out.outputs])
                                skeletons[batch_idx].append(skeletons_list[batch_idx][sk_idx])
                    
                    for i in range(len(batch)):
                        if not sol_texts[i]:
                            sol_texts[i] = [""]
                        if not skeletons[i]:
                            skeletons[i] = [""]
                    
                    # Save results
                    for i, t in enumerate(batch):
                        record = {
                            "global_id": global_idx,
                            "original_id": t["unique_id"],
                            "prompt": t["question"],
                            "skeleton": skeletons[i],
                            "skeleton_prompt": sk_prompts[i],
                            "skeleton_lang": args.skeleton_lang,
                            "responses": sol_texts[i],
                            "question_language": t["lang"],
                            "difficulty": t["difficulty"],
                            "answer": t["answer"],
                            "method": "skeleton_multiturn",
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
    print(f"🌍 Skeleton Language: {args.skeleton_lang}")
    
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    
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
    
    eff_temp = 0.7 if args.rollout > 1 else args.temp
    eff_top_p = 0.95 if args.rollout > 1 else args.top_p
    
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
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    load_targets = []
    if config["type"] == "polymath":
        for l in langs:
            for d in diffs:
                load_targets.append({"lang": l, "split": d, "name": l})
    elif config["type"] in ["math500", "aime25"]:
        for l in langs:
            load_targets.append({"lang": l, "split": l, "name": None})
    elif config["type"] == "mgsm":
        # MGSM requires config name (language code) and uses 'test' split
        for l in langs:
            load_targets.append({"lang": l, "split": "test", "name": l})
    else:
        for d in diffs:
            load_targets.append({"lang": "en", "split": d, "name": None})
    
    if args.method == "skeleton_multiturn":
        run_skeleton_multiturn(args, llm, tok, config, gen_sp, sk_sp, load_targets, langs)

if __name__ == "__main__":
    main()
