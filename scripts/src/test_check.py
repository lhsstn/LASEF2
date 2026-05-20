import json
import os
import requests

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY is not set.")
        return
        
    headers = {"Authorization": f"Bearer {api_key}"}
    
    meta_path = "data/batch_requests/skeleton_translation/batch_jobs.json"
    if not os.path.exists(meta_path):
        print("❌ Metadata file not found.")
        return
        
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    for lang, info in meta.items():
        batch_id = info.get("batch_id")
        if not batch_id:
            continue
            
        print(f"\n==================== Lang: {lang} ====================")
        print(f"Batch ID: {batch_id}")
        
        # 1. Query Batch Status
        res = requests.get(f"https://api.openai.com/v1/batches/{batch_id}", headers=headers)
        if res.status_code != 200:
            print(f"❌ Failed to get batch status: {res.text}")
            continue
            
        batch_info = res.json()
        print("OpenAI Batch API Response:")
        print(json.dumps(batch_info, indent=2))
        
        # 2. Check if output_file_id is null and error_file_id exists
        error_file_id = batch_info.get("error_file_id")
        if error_file_id:
            print(f"\n⚠️ Downloading first few lines of error file: {error_file_id}")
            err_res = requests.get(f"https://api.openai.com/v1/files/{error_file_id}/content", headers=headers)
            if err_res.status_code == 200:
                # Print first 3 lines of error content
                lines = err_res.text.strip().split('\n')
                for i, line in enumerate(lines[:3]):
                    print(f"  Err Line {i+1}: {line}")
            else:
                print(f"  ❌ Failed to download error file content: {err_res.text}")

        # 3. Check if output_file_id exists
        output_file_id = batch_info.get("output_file_id")
        if output_file_id:
            print(f"\n✅ Downloading first few lines of output file: {output_file_id}")
            out_res = requests.get(f"https://api.openai.com/v1/files/{output_file_id}/content", headers=headers)
            if out_res.status_code == 200:
                lines = out_res.text.strip().split('\n')
                for i, line in enumerate(lines[:3]):
                    print(f"  Out Line {i+1}: {line}")
            else:
                print(f"  ❌ Failed to download output file content: {out_res.text}")

if __name__ == "__main__":
    main()
