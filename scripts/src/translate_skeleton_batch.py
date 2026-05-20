import argparse
import json
import os
import sys
import requests

def get_lang_name(lang_code: str) -> str:
    names = {
        "zh": "Chinese",
        "es": "Spanish",
        "ko": "Korean",
        "th": "Thai",
        "ru": "Russian"
    }
    return names.get(lang_code, lang_code)

def create_requests(input_file: str, output_dir: str, target_langs: list, model: str):
    print(f"📖 Reading original skeleton file: {input_file}")
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    
    # Load all records
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"✅ Loaded {len(records)} records.")

    batch_meta = {}
    meta_file = os.path.join(output_dir, "batch_jobs.json")
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                batch_meta = json.load(f)
        except Exception:
            pass

    for lang in target_langs:
        lang_name = get_lang_name(lang)
        req_file = os.path.join(output_dir, f"requests_{lang}.jsonl")
        print(f"✍️ Generating requests for language: {lang_name} -> {req_file}")
        
        count = 0
        with open(req_file, 'w', encoding='utf-8') as f_out:
            for rec in records:
                global_id = rec.get("global_id")
                if global_id is None:
                    continue
                
                skeleton_list = rec.get("skeleton", [])
                if not skeleton_list or not skeleton_list[0]:
                    continue
                
                skeleton_text = skeleton_list[0]
                
                # Build request payload
                custom_id = f"{global_id}_{lang}"
                
                system_prompt = "You are a professional translator specializing in mathematical and scientific terminology."
                user_prompt = (
                    f"Translate the following mathematical/scientific reasoning skeleton into {lang_name}. "
                    "Preserve the markdown structure (**bold**, bullet points). "
                    "Keep mathematical terms, variable names, formulas, and numbers unchanged.\n\n"
                    f"Skeleton:\n{skeleton_text}"
                )
                
                body_data = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                # gpt-5-mini or reasoning models do not support custom temperature (only default 1 is allowed)
                if "gpt-5" not in model.lower() and "o1" not in model.lower() and "o3" not in model.lower():
                    body_data["temperature"] = 0.3
                
                req_data = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body_data
                }
                
                f_out.write(json.dumps(req_data, ensure_ascii=False) + "\n")
                count += 1
        
        print(f"   Generated {count} requests for {lang_name}.")
        batch_meta[lang] = {
            "status": "created",
            "request_file": req_file,
            "target_lang": lang,
            "target_lang_name": lang_name,
            "count": count
        }

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(batch_meta, f, indent=2, ensure_ascii=False)
    print(f"🎉 Batch metadata saved to {meta_file}")

def get_headers():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {api_key}"
    }

def submit_batches(batch_dir: str):
    meta_file = os.path.join(batch_dir, "batch_jobs.json")
    if not os.path.exists(meta_file):
        print(f"❌ Metadata file not found in {batch_dir}. Please create requests first.")
        sys.exit(1)
        
    with open(meta_file, 'r', encoding='utf-8') as f:
        batch_meta = json.load(f)

    headers = get_headers()
    
    for lang, info in batch_meta.items():
        status = info.get("status")
        if status in ["submitted", "completed", "failed"]:
            print(f"⚠️ Batch for {lang} is already in state: {status}. Skipping.")
            continue
            
        req_file = info.get("request_file")
        if not req_file or not os.path.exists(req_file):
            print(f"❌ Request file {req_file} not found. Skipping {lang}.")
            continue
            
        print(f"🚀 Uploading request file for {lang} ({info['target_lang_name']})...")
        
        # 1. Upload File
        upload_url = "https://api.openai.com/v1/files"
        try:
            with open(req_file, 'rb') as f_req:
                files = {
                    "file": (os.path.basename(req_file), f_req, "application/jsonl"),
                    "purpose": (None, "batch")
                }
                res = requests.post(upload_url, headers=headers, files=files)
                res.raise_for_status()
                file_info = res.json()
                file_id = file_info["id"]
                print(f"   Uploaded! File ID: {file_id}")
        except Exception as e:
            print(f"❌ Failed to upload file for {lang}: {e}")
            if 'res' in locals():
                print(res.text)
            continue

        # 2. Submit Batch
        print(f"🚀 Submitting batch job for {lang}...")
        batch_url = "https://api.openai.com/v1/batches"
        batch_payload = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
        
        try:
            res = requests.post(batch_url, headers=headers, json=batch_payload)
            res.raise_for_status()
            batch_info = res.json()
            batch_id = batch_info["id"]
            print(f"   Submitted! Batch ID: {batch_id}")
            
            info["status"] = "submitted"
            info["file_id"] = file_id
            info["batch_id"] = batch_id
        except Exception as e:
            print(f"❌ Failed to submit batch for {lang}: {e}")
            if 'res' in locals():
                print(res.text)
            continue

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(batch_meta, f, indent=2, ensure_ascii=False)
    print(f"🎉 Updated metadata saved to {meta_file}")

def check_batches(batch_dir: str):
    meta_file = os.path.join(batch_dir, "batch_jobs.json")
    if not os.path.exists(meta_file):
        print(f"❌ Metadata file not found in {batch_dir}.")
        sys.exit(1)
        
    with open(meta_file, 'r', encoding='utf-8') as f:
        batch_meta = json.load(f)

    headers = get_headers()
    updated = False
    
    print("\n📊 Batch Job Status:")
    print(f"{'Lang':<6} | {'Language Name':<15} | {'Batch ID':<30} | {'Status':<15} | {'Progress':<15}")
    print("-" * 90)

    for lang, info in batch_meta.items():
        batch_id = info.get("batch_id")
        if not batch_id:
            print(f"{lang:<6} | {info['target_lang_name']:<15} | {'N/A':<30} | {'Not Submitted':<15} | {'-':<15}")
            continue

        url = f"https://api.openai.com/v1/batches/{batch_id}"
        try:
            res = requests.get(url, headers=headers)
            res.raise_for_status()
            batch_info = res.json()
            status = batch_info["status"]
            
            # Progress calculation if available
            req_counts = batch_info.get("request_counts", {})
            total = req_counts.get("total", 0)
            completed = req_counts.get("completed", 0)
            failed = req_counts.get("failed", 0)
            progress_str = f"{completed + failed}/{total}" if total > 0 else "-"
            
            print(f"{lang:<6} | {info['target_lang_name']:<15} | {batch_id:<30} | {status:<15} | {progress_str:<15}")
            
            # Update meta info
            if info["status"] != status or info.get("output_file_id") != batch_info.get("output_file_id"):
                info["status"] = status
                if batch_info.get("output_file_id"):
                    info["output_file_id"] = batch_info["output_file_id"]
                if batch_info.get("error_file_id"):
                    info["error_file_id"] = batch_info["error_file_id"]
                updated = True
        except Exception as e:
            print(f"{lang:<6} | {info['target_lang_name']:<15} | {batch_id:<30} | {f'Error ({str(e)})':<15} | {'-':<15}")
            
    if updated:
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(batch_meta, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Updated statuses in {meta_file}")

def download_batches(batch_dir: str, output_dir: str):
    meta_file = os.path.join(batch_dir, "batch_jobs.json")
    if not os.path.exists(meta_file):
        print(f"❌ Metadata file not found in {batch_dir}.")
        sys.exit(1)
        
    with open(meta_file, 'r', encoding='utf-8') as f:
        batch_meta = json.load(f)

    headers = get_headers()
    os.makedirs(output_dir, exist_ok=True)
    
    for lang, info in batch_meta.items():
        status = info.get("status")
        output_file_id = info.get("output_file_id")
        error_file_id = info.get("error_file_id")
        
        if status == "completed" and not output_file_id:
            print(f"❌ Batch for {lang} completed but FAILED (all requests failed).")
            if error_file_id:
                print(f"   Error File ID: {error_file_id}. You can query its error content using check or manually.")
            continue
            
        if status != "completed" or not output_file_id:
            print(f"⏳ Batch for {lang} is not completed (current status: {status}). Skipping.")
            continue
            
        dest_file = os.path.join(output_dir, f"batch_result_{lang}.jsonl")
        if os.path.exists(dest_file):
            print(f"✅ Result for {lang} already downloaded at {dest_file}.")
            continue
            
        print(f"📥 Downloading results for {lang} (File ID: {output_file_id})...")
        url = f"https://api.openai.com/v1/files/{output_file_id}/content"
        
        try:
            res = requests.get(url, headers=headers)
            res.raise_for_status()
            
            with open(dest_file, 'w', encoding='utf-8') as f_out:
                f_out.write(res.text)
            print(f"   Saved to {dest_file}!")
        except Exception as e:
            print(f"❌ Failed to download results for {lang}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage OpenAI Batch API skeleton translations")
    
    # Mode flags
    parser.add_argument("--create", action="store_true", help="Create batch request files")
    parser.add_argument("--submit", action="store_true", help="Submit created batch files to OpenAI")
    parser.add_argument("--check", action="store_true", help="Check status of submitted batch jobs")
    parser.add_argument("--download", action="store_true", help="Download completed batch results")
    
    # Arguments
    parser.add_argument("--input", type=str, default="data/results/SkelLang/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl")
    parser.add_argument("--batch_dir", type=str, default="data/batch_requests/skeleton_translation")
    parser.add_argument("--target_langs", type=str, default="zh,es,ko,th,ru", help="Comma separated target languages")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--download_dir", type=str, default="data/batch_requests/skeleton_translation/results")
    
    args = parser.parse_args()
    
    if not (args.create or args.submit or args.check or args.download):
        parser.print_help()
        sys.exit(1)
        
    if args.create:
        langs = [l.strip() for l in args.target_langs.split(",") if l.strip()]
        create_requests(args.input, args.batch_dir, langs, args.model)
        
    if args.submit:
        submit_batches(args.batch_dir)
        
    if args.check:
        check_batches(args.batch_dir)
        
    if args.download:
        download_batches(args.batch_dir, args.download_dir)

if __name__ == "__main__":
    main()
