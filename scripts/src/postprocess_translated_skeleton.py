import argparse
import json
import os
import sys

def postprocess(original_path: str, batch_result_path: str, target_lang: str, output_path: str):
    print(f"📖 Reading original skeleton file: {original_path}")
    if not os.path.exists(original_path):
        print(f"❌ Original file not found: {original_path}")
        sys.exit(1)
        
    print(f"📥 Reading batch result file: {batch_result_path}")
    if not os.path.exists(batch_result_path):
        print(f"❌ Batch result file not found: {batch_result_path}")
        sys.exit(1)

    # 1. Parse batch results and map by global_id
    translations = {}
    error_count = 0
    success_count = 0

    with open(batch_result_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                custom_id = data.get("custom_id")
                if not custom_id:
                    continue
                
                # custom_id format: "global_id_lang" (e.g., "123_zh")
                parts = custom_id.rsplit('_', 1)
                if len(parts) != 2:
                    print(f"⚠️ Unexpected custom_id format: {custom_id}")
                    continue
                
                global_id_str, lang = parts
                global_id = int(global_id_str)
                
                error = data.get("error")
                if error:
                    print(f"⚠️ Error in batch response for custom_id {custom_id}: {error}")
                    error_count += 1
                    continue
                
                response = data.get("response")
                if not response or response.get("status_code") != 200:
                    status_code = response.get("status_code") if response else "N/A"
                    print(f"⚠️ Invalid status code {status_code} for custom_id {custom_id}")
                    error_count += 1
                    continue
                
                body = response.get("body", {})
                choices = body.get("choices", [])
                if not choices:
                    print(f"⚠️ No choices in response body for custom_id {custom_id}")
                    error_count += 1
                    continue
                
                translated_text = choices[0].get("message", {}).get("content", "").strip()
                if not translated_text:
                    print(f"⚠️ Empty content for custom_id {custom_id}")
                    error_count += 1
                    continue
                
                translations[global_id] = translated_text
                success_count += 1
            except Exception as e:
                print(f"⚠️ Failed to parse line in batch result: {e}")
                error_count += 1

    print(f"✅ Loaded {success_count} translations. Failed: {error_count}")

    # 2. Update original skeleton records and write to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_records = 0
    translated_records = 0
    fallback_records = 0

    with open(original_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            try:
                rec = json.loads(line)
                global_id = rec.get("global_id")
                
                if global_id is not None and global_id in translations:
                    # Replace skeleton with translated one
                    rec["skeleton"] = [translations[global_id]]
                    rec["skeleton_lang"] = target_lang
                    translated_records += 1
                else:
                    # Fallback to original skeleton (or keep as is) but flag it
                    rec["skeleton_lang"] = rec.get("skeleton_lang", "en")
                    fallback_records += 1
                
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_records += 1
            except Exception as e:
                print(f"⚠️ Error writing updated record: {e}")

    print(f"🎉 Done! Postprocessed {total_records} records.")
    print(f"   Successfully translated: {translated_records} ({translated_records/total_records*100:.2f}%)")
    print(f"   Fallback to original: {fallback_records}")
    print(f"   Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Postprocess GPT Batch translation results and generate translated skeleton files")
    parser.add_argument("--original", type=str, required=True, help="Path to original English skeleton JSONL file")
    parser.add_argument("--batch_result", type=str, required=True, help="Path to GPT Batch result JSONL file")
    parser.add_argument("--target_lang", type=str, required=True, help="Target language code (e.g. zh, es, ko, th, ru)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the postprocessed skeleton JSONL file")
    
    args = parser.parse_args()
    postprocess(args.original, args.batch_result, args.target_lang, args.output)

if __name__ == "__main__":
    main()
