#!/usr/bin/env python3
"""
搜索正确的数据集仓库名称
"""

from huggingface_hub import HfApi

api = HfApi()

# 搜索数据集
datasets_to_search = [
    "MATH-500",
    "MMLU-Pro",
    "Omni-MATH", 
    "TheoremQA"
]

print("搜索数据集仓库...")
print("=" * 60)

for dataset_name in datasets_to_search:
    print(f"\n搜索: {dataset_name}")
    try:
        # 搜索数据集
        results = api.list_datasets(search=dataset_name, limit=5)
        print(f"找到 {len(results)} 个结果:")
        for dataset in results:
            print(f"  - {dataset.id} (作者: {dataset.author}, 下载量: {dataset.downloads})")
    except Exception as e:
        print(f"搜索失败: {e}")

print("\n" + "=" * 60)
print("手动搜索建议:")
print("1. MATH-500: 可能是 'EleutherAI/math-500' 或 'cais/mmlu'")
print("2. MMLU-Pro: 已成功下载 TIGER-Lab/MMLU-Pro")
print("3. Omni-MATH: 可能是 'omni-math/omni-math' 或其他名称")
print("4. TheoremQA: 可能是 'wenh06/TheoremQA' 或其他名称")