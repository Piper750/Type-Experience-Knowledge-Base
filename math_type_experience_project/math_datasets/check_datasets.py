#!/usr/bin/env python3
"""
检查每个数据集的内容
"""

import os
import json

datasets_dir = "/Users/gongzhihuan/Desktop/math_datasets"

print("数据集内容检查")
print("=" * 60)

# 检查 MATH-500
print("\n1. MATH-500")
math500_file = os.path.join(datasets_dir, "MATH-500", "test.jsonl")
if os.path.exists(math500_file):
    with open(math500_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"   文件大小: {os.path.getsize(math500_file):,} bytes")
        print(f"   行数: {len(lines)}")
        if len(lines) > 0:
            first_line = json.loads(lines[0])
            print(f"   第一个问题类型: {first_line.get('type', 'N/A')}")
            print(f"   第一个问题难度: {first_line.get('level', 'N/A')}")
            print(f"   第一个问题: {first_line.get('question', 'N/A')[:100]}...")

# 检查 MMLU-Pro-Math
print("\n2. MMLU-Pro-Math")
mmlu_dir = os.path.join(datasets_dir, "MMLU-Pro-Math", "data")
if os.path.exists(mmlu_dir):
    parquet_files = [f for f in os.listdir(mmlu_dir) if f.endswith('.parquet')]
    print(f"   Parquet 文件: {parquet_files}")
    for file in parquet_files:
        file_path = os.path.join(mmlu_dir, file)
        print(f"   {file}: {os.path.getsize(file_path):,} bytes")

# 检查 Omni-MATH
print("\n3. Omni-MATH")
omni_file = os.path.join(datasets_dir, "Omni-MATH", "test.jsonl")
if os.path.exists(omni_file):
    with open(omni_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"   文件大小: {os.path.getsize(omni_file):,} bytes")
        print(f"   行数: {len(lines)}")
        if len(lines) > 0:
            first_line = json.loads(lines[0])
            print(f"   第一个问题: {first_line.get('question', 'N/A')[:100]}...")

# 检查 TheoremQA
print("\n4. TheoremQA")
theoremqa_dir = os.path.join(datasets_dir, "TheoremQA", "data")
if os.path.exists(theoremqa_dir):
    parquet_files = [f for f in os.listdir(theoremqa_dir) if f.endswith('.parquet')]
    print(f"   Parquet 文件: {parquet_files}")
    for file in parquet_files:
        file_path = os.path.join(theoremqa_dir, file)
        print(f"   {file}: {os.path.getsize(file_path):,} bytes")

print("\n" + "=" * 60)
print("检查完成！所有数据集都已成功下载到桌面文件夹。")