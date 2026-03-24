#!/usr/bin/env python3
"""
搜索数据集的正确仓库名称
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
        results = list(api.list_datasets(search=dataset_name, limit=10))
        print(f"找到 {len(results)} 个结果:")
        for i, dataset in enumerate(results[:5]):  # 只显示前5个
            print(f"  {i+1}. {dataset.id}")
            if hasattr(dataset, 'description') and dataset.description:
                print(f"     描述: {dataset.description[:100]}...")
    except Exception as e:
        print(f"搜索失败: {e}")

print("\n" + "=" * 60)
print("搜索完成！")