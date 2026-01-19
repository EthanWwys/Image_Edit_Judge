import os
import json
import argparse
import re
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Build Image Editing Testset")
    parser.add_argument("--mode", type=str, required=True, choices=['drone', 'egovid', 'walk'], help="Dataset mode")
    parser.add_argument("--source_json", type=str, required=True, help="Path to original metadata JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory for verification (Egovid) or Ignored")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output testset JSON")
    parser.add_argument("--filter_ids", type=str, default=None, help="Optional: Comma separated IDs to filter")
    return parser.parse_args()

def load_source_data(json_path):
    print(f"📖 Loading source data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 如果是列表，转为 dict map；如果是 dict，直接使用
        if isinstance(data, list):
            return {item['id']: item for item in data}
        else:
            return data
    except Exception as e:
        print(f"❌ Error loading source JSON: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    source_map = load_source_data(args.source_json)
    
    testset = []
    target_ids = set(args.filter_ids.split(',')) if args.filter_ids else None
    
    print(f"📂 Current Working Directory: {os.getcwd()}")
    
    # ==========================================
    # 模式 A: Drone / Walk (原有逻辑：正则匹配 key)
    # ==========================================
    if args.mode in ['drone', 'walk']:
        print(f"🚀 Mode [{args.mode}]: Iterating JSON with Regex Matching")
        
        total_keys_found = 0
        files_missing = 0
        
        for original_id, item in source_map.items():
            if target_ids and original_id not in target_ids:
                continue

            json_path = item.get('last_frame_path', '')
            if not json_path: continue

            pattern = re.compile(r'^SC\d+_MOD_\d+$')

            for key, prompt_text in item.items():
                if pattern.match(key):
                    total_keys_found += 1
                    full_path = os.path.join(json_path, f"{key}.jpg")
                    
                    if os.path.exists(full_path):
                        testset.append({
                            "test_id": f"{original_id}_{key}",
                            "original_id": original_id,
                            "prompt": prompt_text,
                            "prompt_key": key,
                            "last_frame_path": full_path,
                            "first_frame_path": item.get('first_frame_path'),
                            "mode": args.mode
                        })
                    else:
                        files_missing += 1
                        if files_missing <= 3:
                            print(f"❌ [Missing] {full_path}")

        print(f"   - Total keys processed: {total_keys_found}")
        print(f"   - Files missing: {files_missing}")

    # ==========================================
    # 模式 B: Egovid (修改后：直接读取 JSON 路径)
    # ==========================================
    elif args.mode == 'egovid':
        print(f"🚀 Mode [{args.mode}]: Iterating JSON and verifying last_frame_path")
        
        files_missing = 0
        
        for original_id, item in source_map.items():
            if target_ids and original_id not in target_ids:
                continue
            
            # 1. 获取路径
            # 根据提供的 metadata，路径直接就在 last_frame_path 字段里
            relative_path = item.get('last_frame_path')
            
            if not relative_path:
                continue
                
            # 2. 验证文件是否存在
            # 假设 metadata 中的路径是相对于运行目录的 (如 results/exp_unified/...)
            if os.path.exists(relative_path):
                # 3. 获取 Prompt (优先取 lf_prompt_v4_minimal)
                prompt = item.get('lf_prompt_v4_minimal')
                if not prompt:
                    # 如果 minimal 没有，尝试 fallback 到 instruction 或其他字段
                    prompt = item.get('instruction', '')
                
                testset.append({
                    "test_id": original_id, # Egovid ID 本身就是唯一的
                    "original_id": original_id,
                    "prompt": prompt,
                    "prompt_key": "lf_prompt_v4_minimal",
                    "last_frame_path": relative_path,
                    "first_frame_path": item.get('first_frame_path'),
                    "mode": args.mode
                })
            else:
                files_missing += 1
                if files_missing <= 3:
                    print(f"❌ [Missing] JSON path not found on disk: {relative_path}")
        
        print(f"   - Files missing: {files_missing}")

    # ==========================================
    # 输出 JSON 文件 (Dataset Output)
    # ==========================================
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(testset, f, indent=4, ensure_ascii=False)
        
    print(f"✅ JSON Build Complete! Valid items: {len(testset)}")
    print(f"💾 Saved Dataset to: {args.output_path}")

    # ==========================================
    # 输出 Log 文件 (JSON 格式)
    # ==========================================
    log_dir = os.path.join("results", "exp_unified", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{args.mode}.json"
    log_path = os.path.join(log_dir, log_filename)

    print(f"📝 Generating Log file: {log_path} ...")
    
    log_data = {
        "mode": args.mode,
        "total_count": len(testset),
        "items": []
    }

    for item in testset:
        log_data["items"].append({
            "test_id": item.get('test_id'),
            "last_frame_path": item.get('last_frame_path')
        })
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f_log:
            json.dump(log_data, f_log, indent=4, ensure_ascii=False)
        print(f"✅ Log Saved to: {log_path}")
    except Exception as e:
        print(f"❌ Error writing log file: {e}")

if __name__ == "__main__":
    main()