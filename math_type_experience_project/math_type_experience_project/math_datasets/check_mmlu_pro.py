#!/usr/bin/env python3
"""
检查 MMLU-Pro 数据集内容
"""

import pandas as pd
import os

# 读取 Parquet 文件
parquet_file = "/Users/gongzhihuan/Desktop/math_datasets/MMLU-Pro-Math/data/test-00000-of-00001.parquet"

if os.path.exists(parquet_file):
    print(f"读取文件: {parquet_file}")
    
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file)
    
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n前几行数据:")
    print(df.head())
    
    # 检查是否有数学相关的题目
    if 'subject' in df.columns:
        math_subjects = df[df['subject'].str.contains('math', case=False, na=False)]
        print(f"\n数学相关题目数量: {len(math_subjects)}")
        if len(math_subjects) > 0:
            print("数学题目示例:")
            print(math_subjects.head())
    
    # 检查问题类型
    if 'question' in df.columns:
        print(f"\n问题示例:")
        print(df['question'].iloc[0] if len(df) > 0 else "无数据")
else:
    print(f"文件不存在: {parquet_file}")