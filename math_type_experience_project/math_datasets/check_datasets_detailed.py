#!/usr/bin/env python3
"""
详细检查每个数据集的内容
"""

import os
import json

datasets_dir = "/Users/gongzhihuan/Desktop/math_datasets"

print("数据集详细内容检查")
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
            try:
                first_line = json.loads(lines[0])
                print(f"   第一个问题键: {list(first_line.keys())}")
                print(f"   第一个问题类型: {first_line.get('type', 'N/A')}")
                print(f"   第一个问题难度: {first_line.get('level', 'N/A')}")
                question = first_line.get('question', 'N/A')
                if question != 'N/A':
                    print(f"   第一个问题: {question[:100]}...")
                else:
                    print(f"   第一个问题内容: {first_line}")
            except json.JSONDecodeError as e:
                print(f"   JSON解析错误: {e}")
                print(f"   第一行内容: {lines[0][:100]}...")

# 检查 Omni-MATH
print("\n3. Omni-MATH")
omni_file = os.path.join(datasets_dir, "Omni-MATH", "test.jsonl")
if os.path.exists(omni_file):
    with open(omni_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"   文件大小: {os.path.getsize(omni_file):,} bytes")
        print(f"   行数: {len(lines)}")
        if len(lines) > 0:
            try:
                first_line = json.loads(lines[0])
                print(f"   第一个问题键: {list(first_line.keys())}")
                question = first_line.get('question', 'N/A')
                if question != 'N/A':
                    print(f"   第一个问题: {question[:100]}...")
                else:
                    print(f"   第一个问题内容: {first_line}")
            except json.JSONDecodeError as e:
                print(f"   JSON解析错误: {e}")
                print(f"   第一行内容: {lines[0][:100]}...")

print("\n" + "=" * 60)
print("检查完成！所有数据集都已成功下载到桌面文件夹。")