import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import json
import yaml
import math
import time
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor
import torch.multiprocessing as mp

# try:
#     mp.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass

from vllm import LLM, SamplingParams
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default=None)
args, _ = parser.parse_known_args()

# ===================== 配置加载 =====================
def load_config(config_path="./config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
if args.mode:
    INPUT_MODE = args.mode.lower()
else:
    INPUT_MODE = CONFIG['paths'].get('input_mode', 'egovid').lower()
    base_json = CONFIG['paths']['json_file']
base_json = CONFIG['paths']['json_file']
if INPUT_MODE in ['drone', 'walk']:
    # 假设 base_json 是 metadata.json，这里替换为 drone.json / walk.json
    JSON_PATH = os.path.join(os.path.dirname(base_json), f"{INPUT_MODE}.json")
else:
    JSON_PATH = base_json
MODEL_PATH = CONFIG['models']['vlm_path']
RESULT_ROOT = CONFIG['paths']['result_root']

# ===================== [STRICT] System Prompts =====================

# --- 1. EGOVID Prompt (From image_edit_1/testset_prompt.py) ---
SYSTEM_PROMPT_EGOVID = """
Role: You are a professional Image Editing Prompt Expert.
Task: Generate a MINIMAL and PRECISE image editing prompt based on the Start Frame and the Instruction.

Goal: Translate the process-oriented "Instruction" into a result-oriented "Edit Command" that describes the final state. The change should be visible but minimal to ensure smooth video generation.

Steps:
1. **Caption**: Briefly describe the current state of the main object/hand in the Start Frame.
2. **Analyze**: Predict the end state of the instruction. Compare the caption with the Instruction. Identify the action that is expected to be completed in the image.
3. **Rewrite**: Convert the uncompleted action into a concise edit command (e.g., "add X", "move Y to Z", "change state of A").
4. **Constraint**: Explicitly state what should remain unchanged (background, lighting, objects).

### Example:
[Start Frame]: (A hand holding a watermelon next to a mesh bag on the ground.)
[Instruction]: holding watermelon, placing watermelon in mesh bag, adjusting bag.
[Analysis]: The image shows the "holding" phase. The "placing in bag" is the next step.
[Edit Command]: Place the watermelon inside the white mesh bag. The hand is now adjusting the bag opening. The soil and grass background remains unchanged.

### Now process this:
[Start Frame]: (The image provided above)
[Instruction]: {instruction}
[Edit Command]:"""

# --- 2. DRONE Prompts (From image_edit_2/prompt.py) ---
SC1_BATCH_TEMPLATE = """Task: Generate {count} distinct spatial camera movement commands for [SC1: Camera Motion].
[Role]: Professional Cinematic Narrative Director.
[Goal]: Describe a SIGNIFICANT camera shift and its precise impact on the composition based on the provided image.

[STRICT RULES]:
1. COMPOSITION FEEDBACK: You MUST describe how the objects in the current image move within the frame and what new elements appear.
2. LANGUAGE: Output strictly in English.
3. FORMAT: Each MOD must follow this structure: "The camera has [action] significantly by [magnitude]. The [original object] originally at the [original position] has now moved towards the [opposite direction of camera action] to the [new position], revealing [new content] at the [action direction]. Lighting and terrain remain consistent."
4. SINGLE DIMENSION: Focus on ONE major movement per MOD (Yaw, Pitch, or Translation).
5. MAGNITUDE: Use specific degrees (e.g., 30°, 60°, 120°). 
6. PERSPECTIVE REQUIREMENT: MOD_1 must adopt a bird’s-eye view, and its camera action must align with this perspective.

Output strictly in JSON format:
{{
  "MOD_1": "...", "MOD_2": "...", "MOD_3": "...", "MOD_4": "...", "MOD_5": "...", "MOD_6": "..."
}}"""

# ===================== SC2: Background Transition (Original) =====================
SC2_BATCH_TEMPLATE = """Task: Write {count} background transition commands for [SC2].
[Role]: Realistic Environment Concept Artist.
[Goal]: Change ONLY the geographic setting/location. Keep the main subject and weather IDENTICAL to the original image.

[STRICT RULES]:
1. SUBJECT RETENTION: Identify the primary subject (person, vehicle, animal, or specific structure). 
   - CRITICAL: If no clear central subject exists, you MUST treat the existing composition's perspective, horizon line, and overall layout as the "Subject". Preserve the structural skeleton of the image.
2. PURE SETTING: Replace only the terrain or environment type (e.g., from green hills to a rocky desert). Do not relocate or remove the main objects.
3. WEATHER CONSISTENCY: Do NOT change the weather, time of day, or lighting. If the original is sunny, the new setting must also be sunny.
4. REALISM ONLY: Use "Realistic Modern City", "Natural Landscape", etc. Strictly AVOID "Cyberpunk", "Futuristic", or "Sci-fi".
5. NO ATMOSPHERE: Do not use words like "rainy", "snowy", "stormy", or "misty". Focus on "Location" only.
6. FORMAT: Each MOD must follow this structure: "The main subject [main subject name/description] remains unchanged. The original [original setting] is replaced with a [new realistic setting], preserving the same weather, lighting, and composition layout. The terrain transitions seamlessly while keeping the subject’s position and proportions consistent."

Output strictly in JSON format:
{{
  "MOD_1": "...", "MOD_2": "...", "MOD_3": "...", "MOD_4": "...", "MOD_5": "..."
}}"""

# ===================== SC4: Dynamic Activity (New) =====================
SC4_BATCH_TEMPLATE = """Task: Generate {count} distinct dynamic activity commands for [SC-4].
Goal: Based on the provided starting frame, describe a 5-second realistic movement of the {object} that maintains high visual consistency with the original scene.

[STRICT RULES]:
1. IMAGE FIDELITY & PERSISTENCE: The motion must be a direct continuation of the starting frame. Identify the specific agents (people, animals, vehicles) already present in the image and explicitly describe their subsequent actions or trajectory (e.g., "the person standing by the tree starts walking towards the bench").
2. 5-SECOND REALISM: The motion must be achievable within 5 seconds. Describe agents moving a short distance (e.g., "walking a few steps forward", "a car passing through the intersection").
3. PERSPECTIVE: Since this is a walking/pedestrian view, describe agents relative to the camera (e.g., "coming towards the camera", "crossing from left to right").
4. OBJECT DESTINATION: Clearly state where the objects are going or how their state changes (e.g., "the dog runs out of the right frame", "the parked car starts to pull away").
5. NO BLUR/SHARPNESS: Keep agents crisp and sharp as if captured with high shutter speed.
6. CONTINUITY: The background, lighting, and static objects must remain identical to the original image.

{object}: various people | vehicles | people & vehicles | animals | boats | ships
{density}: sparse | moderate
{activity}: calm | normal | busy

Output strictly in JSON format:
{{
  "MOD_1": "...", "MOD_2": "...", "MOD_3": "...", "MOD_4": "..."
}}"""

# ===================== SC5: Lighting/Atmosphere (Original) =====================
SC5_BATCH_TEMPLATE = """Task: Write {count} lighting/atmosphere commands. Change ONLY light/weather, keep geometry identical.
[Consistency]: DO NOT change the geometry or identity of any objects.
Output strictly in JSON format: {{ "MOD_1": "...", "MOD_2": "...", "MOD_3": "...", "MOD_4": "..." , "MOD_5": "...", "MOD_6": "..."}}"""



# ===================== 任务构建逻辑 =====================
def build_tasks(data_items, mode):
    tasks = []
    print(f"🔧 Building tasks for mode: {mode.upper()}")

    for idx, item in enumerate(data_items):
        ff_path = item.get('first_frame_path')
        if not ff_path or not os.path.exists(ff_path):
            continue

        # --- Case 1: EGOVID ---
        if mode == 'egovid':
            if not item.get('lf_prompt_v4_minimal'):
                prompt = SYSTEM_PROMPT_EGOVID.format(instruction=item.get('instruction', ''))
                tasks.append((idx, "lf_prompt_v4_minimal", prompt))

        # --- Case 2: DRONE ---
        elif mode == 'drone':
            # tasks.append((idx, "SC1_BATCH", SC1_BATCH_TEMPLATE.format(count=6)))
            # tasks.append((idx, "SC2_BATCH", SC2_BATCH_TEMPLATE.format(count=5)))
            tasks.append((idx, "SC4_BATCH",
                               SC4_BATCH_TEMPLATE.format(count=3, object="contextual agents", density="various",
                                                         activity="various")))
            # tasks.append((idx, "SC5_BATCH", SC5_BATCH_TEMPLATE.format(count=6)))

        # --- Case 3: WALK ---
        elif mode == 'walk':
            # tasks.append((idx, "SC1_BATCH", SC1_BATCH_TEMPLATE.format(count=6)))
            # tasks.append((idx, "SC2_BATCH", SC2_BATCH_TEMPLATE.format(count=5)))
            tasks.append((idx, "SC4_BATCH",
                               SC4_BATCH_TEMPLATE.format(count=3, object="contextual agents", density="various",
                                                         activity="various")))
            # tasks.append((idx, "SC5_BATCH", SC5_BATCH_TEMPLATE.format(count=6)))
            
    return tasks

# ===================== 主逻辑 =====================
def main():
    if not os.path.exists(JSON_PATH):
        print(f"❌ JSON not found: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_tasks = build_tasks(data, INPUT_MODE)
    
    if not all_tasks:
        print("🎉 No tasks to process.")
        return

    # 初始化 vLLM (显存优化)
    num_gpus = torch.cuda.device_count()
    try:
        llm = LLM(
            model=MODEL_PATH, 
            trust_remote_code=True, 
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image": 1},
            max_model_len=32768
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        return

    batch_size = 50
    sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)
    total_chunks = math.ceil(len(all_tasks) / batch_size)

    print(f"⚡ Starting Inference for {len(all_tasks)} tasks...")

    for i in range(0, len(all_tasks), batch_size):
        chunk_tasks = all_tasks[i : i + batch_size]
        vllm_inputs = []
        
        for (item_idx, field_name, prompt_text) in chunk_tasks:
            try:
                item = data[item_idx]
                image_path = item['first_frame_path']
                image = Image.open(image_path).convert("RGB")
                
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text}
                ]}]
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                vllm_inputs.append({
                    "prompt": text_input,
                    "multi_modal_data": {"image": image}
                })
            except Exception as e:
                print(f"⚠️ Error preparing input: {e}")
                continue

        if vllm_inputs:
            try:
                outputs = llm.generate(vllm_inputs, sampling_params)
                
                for j, output in enumerate(outputs):
                    original_task = chunk_tasks[j]
                    item_idx, field_name, _ = original_task
                    generated_text = output.outputs[0].text.strip()
                    
                    if INPUT_MODE == 'egovid':
                        data[item_idx][field_name] = generated_text
                    else:
                        # JSON 解析 (Drone/Walk)
                        try:
                            start = generated_text.find('{')
                            end = generated_text.rfind('}') + 1
                            if start != -1 and end != -1:
                                json_data = json.loads(generated_text[start:end])
                                prefix = field_name.split('_')[0]
                                for key, val in json_data.items():
                                    full_key = f"{prefix}_{key}"
                                    data[item_idx][full_key] = val
                            else:
                                data[item_idx][f"{field_name}_raw"] = generated_text
                        except Exception:
                            data[item_idx][f"{field_name}_error"] = generated_text

            except Exception as e:
                print(f"❌ Batch Inference Error: {e}")

        # 实时保存
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Batch {i//batch_size + 1}/{total_chunks} Done.")

if __name__ == "__main__":
    main()