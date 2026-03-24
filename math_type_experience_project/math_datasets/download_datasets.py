#!/usr/bin/env python3
"""
下载数学数据集脚本
MATH-500, MMLU-Pro-Math, Omni-MATH, TheoremQA
"""

import os
import json
from huggingface_hub import snapshot_download

# 数据集配置
datasets = {
    "MATH-500": {
        "repo": "HuggingFaceTB/math-500",
        "description": "MATH-500 数据集，包含500个数学问题"
    },
    "MMLU-Pro-Math": {
        "repo": "TIGER-Lab/MMLU-Pro",
        "description": "MMLU-Pro 数学部分数据集"
    },
    "Omni-MATH": {
        "repo": "omni-math/omni-math",
        "description": "Omni-MATH 数学推理数据集"
    },
    "TheoremQA": {
        "repo": "wenh06/TheoremQA",
        "description": "TheoremQA 定理问答数据集"
    }
}

def download_dataset(name, repo, description):
    """下载单个数据集"""
    print(f"\n{'='*60}")
    print(f"下载数据集: {name}")
    print(f"描述: {description}")
    print(f"仓库: {repo}")
    print(f"{'='*60}")

    try:
        # 创建数据集目录
        dataset_dir = os.path.join(os.getcwd(), name)
        os.makedirs(dataset_dir, exist_ok=True)

        # 下载数据集
        print(f"开始下载 {name}...")
        result = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=dataset_dir,
            local_dir_use_symlinks=False
        )

        print(f"✅ {name} 下载完成！")
        print(f"保存位置: {dataset_dir}")
        print(f"文件数量: {len(os.listdir(dataset_dir))}")

        # 创建 README 文件
        readme_path = os.path.join(dataset_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n")
            f.write(f"**描述**: {description}\n\n")
            f.write(f"**HuggingFace 仓库**: {repo}\n\n")
            f.write(f"**下载时间**: {os.path.getctime(dataset_dir)}\n\n")
            f.write(f"**文件列表**:\n")
            for item in os.listdir(dataset_dir):
                f.write(f"- {item}\n")

        return True

    except Exception as e:
        print(f"❌ 下载 {name} 失败: {e}")
        return False

def main():
    """主函数"""
    print("数学数据集下载工具")
    print("=" * 60)

    # 确保在正确的目录
    os.chdir("/Users/gongzhihuan/Desktop/math_datasets")
    print(f"工作目录: {os.getcwd()}")

    # 下载所有数据集
    results = {}
    for name, config in datasets.items():
        success = download_dataset(name, config["repo"], config["description"])
        results[name] = success

    # 总结
    print("\n" + "=" * 60)
    print("下载总结")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")

    # 创建总 README
    total_readme = os.path.join(os.getcwd(), "README.md")
    with open(total_readme, "w", encoding="utf-8") as f:
        f.write("# 数学数据集集合\n\n")
        f.write("本文件夹包含以下数学数据集：\n\n")
        for name, config in datasets.items():
            f.write(f"## {name}\n")
            f.write(f"- **描述**: {config['description']}\n")
            f.write(f"- **HuggingFace 仓库**: {config['repo']}\n")
            f.write(f"- **状态**: {'✅ 已下载' if results.get(name) else '❌ 下载失败'}\n\n")

    print(f"\n总 README 已创建: {total_readme}")

if __name__ == "__main__":
    main()