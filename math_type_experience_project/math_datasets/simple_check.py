#!/usr/bin/env python3
"""
简单检查数据集
"""

import os

# 检查目录结构
math_datasets_dir = "/Users/gongzhihuan/Desktop/math_datasets"

print("数学数据集目录结构:")
print("=" * 60)

for item in os.listdir(math_datasets_dir):
    item_path = os.path.join(math_datasets_dir, item)
    if os.path.isdir(item_path):
        print(f"\n📁 {item}/")
        # 列出子目录和文件
        for subitem in os.listdir(item_path):
            subitem_path = os.path.join(item_path, subitem)
            if os.path.isdir(subitem_path):
                print(f"  📁 {subitem}/")
                # 列出更深层的文件
                for file in os.listdir(subitem_path):
                    file_path = os.path.join(subitem_path, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f"    📄 {file} ({size:,} bytes)")
            else:
                size = os.path.getsize(subitem_path)
                print(f"  📄 {subitem} ({size:,} bytes)")

print("\n" + "=" * 60)
print("总结:")
print("- MMLU-Pro-Math: 已下载 (Parquet 格式)")
print("- 其他数据集: 下载失败，需要正确的仓库名称")