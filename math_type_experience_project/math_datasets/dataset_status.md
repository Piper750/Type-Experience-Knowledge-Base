# 数学数据集下载状态

## 已完成
- ✅ **MMLU-Pro-Math**: 已成功下载
  - 仓库: `TIGER-Lab/MMLU-Pro`
  - 格式: Parquet 文件
  - 位置: `/Users/gongzhihuan/Desktop/math_datasets/MMLU-Pro-Math/`

## 失败
- ❌ **MATH-500**: 下载失败
  - 尝试仓库: `HuggingFaceTB/math-500`, `EleutherAI/math-500`
  - 错误: 404 Not Found

- ❌ **Omni-MATH**: 下载失败
  - 尝试仓库: `omni-math/omni-math`, `omni-math/Omni-MATH`
  - 错误: 404 Not Found

- ❌ **TheoremQA**: 下载失败
  - 尝试仓库: `wenh06/TheoremQA`
  - 错误: 404 Not Found

## 下一步
需要搜索正确的 HuggingFace 仓库名称：
1. MATH-500 数据集的正确仓库
2. Omni-MATH 数据集的正确仓库  
3. TheoremQA 数据集的正确仓库

## 建议
可以尝试以下方法：
1. 在 HuggingFace 官网搜索这些数据集
2. 查看相关论文或 GitHub 仓库的文档
3. 使用 `huggingface-cli search` 命令搜索