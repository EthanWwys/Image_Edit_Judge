#!/bin/bash

# 开启错误检测：如果任何命令返回非零状态（报错），脚本将立即停止执行
set -e

echo "===drone, walk的SC4重新生成"
python test_prompt_vllm.py.py --mode drone
python test_prompt_vllm.py.py --mode walk

python test_lastframe_gen.py --mode drone
python test_lastframe_gen.py --mode walk

echo "=== 开始执行 build.py 任务 ==="

# 1. Build: egovid
echo "[1/6] Processing egovid build..."
python build.py \
  --mode egovid \
  --source_json metadata.json \
  --image_dir results/exp_unified/egovid/generated_frames \
  --output_path ./egovid_metadata.json

# 2. Build: drone
echo "[2/6] Processing drone build..."
python build.py \
  --mode drone \
  --source_json drone.json \
  --image_dir results/exp_unified/drone/generated_frames \
  --output_path ./drone_metadata.json

# 3. Build: walk
echo "[3/6] Processing walk build..."
python build.py \
  --mode walk \
  --source_json walk.json \
  --image_dir results/exp_unified/walk/generated_frames \
  --output_path ./walk_metadata.json


echo "=== 开始执行 vlm_judge.py 任务 ==="

# 4. Judge: egovid
echo "[4/6] Judging egovid..."
python vlm_judge.py \
  --input_json egovid_metadata.json \
  --config config.yaml

# 5. Judge: drone
echo "[5/6] Judging drone..."
python vlm_judge.py \
  --input_json drone_metadata.json \
  --config config.yaml

# 6. Judge: walk
echo "[6/6] Judging walk..."
python vlm_judge.py \
  --input_json walk_metadata.json \
  --config config.yaml


echo "=== 开始执行文件归档 ==="

# 定义目标文件夹
DEST_DIR="results"

# 检查目录是否存在，不存在则创建
if [ ! -d "$DEST_DIR" ]; then
    echo "创建目录: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# 复制评测结果文件
# 假设 vlm_judge.py 生成的文件名为 *_judged_direct.json
cp egovid_metadata_judged_direct.json "$DEST_DIR/"
cp drone_metadata_judged_direct.json "$DEST_DIR/"
cp walk_metadata_judged_direct.json "$DEST_DIR/"

# 复制原始 Source JSON 文件
cp metadata.json "$DEST_DIR/"
cp drone.json "$DEST_DIR/"
cp walk.json "$DEST_DIR/"

echo "所有任务已完成，文件已归档至 $DEST_DIR 目录。"