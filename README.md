# TE-KB 数学推理实验

这是一个围绕“题型—经验知识库（Type-Experience Knowledge Base）”构建的可运行实验工程，目标是把你的研究方案变成一套真正能执行的离线建库 + 在线检索增强推理流水线。

## 目录结构

```text
math_type_experience_project/
├── configs/
│   └── default.yaml
├── scripts/
│   ├── build_kb.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   └── smoke_test.py
├── src/
│   ├── dataset.py
│   ├── evaluation.py
│   ├── heuristics.py
│   ├── io_utils.py
│   ├── kb_builder.py
│   ├── llm_backends.py
│   ├── pipeline.py
│   ├── retriever.py
│   └── schema.py
├── outputs/
└── requirements.txt
```

## 代码实现了什么

### 1. 离线知识库构建
- 自动读取 `../math_datasets` 下的 `jsonl/json/csv` 文件；
- 对题目做题型抽象（coarse/fine type）；
- 生成结构化经验（步骤、原则、公式、易错点）；
- 计算验证分数（答案有效性、题型一致性、步骤质量、经验匹配）；
- 产出 `kb_entries.jsonl`、`type_taxonomy.json` 与 `retriever.pkl`。

### 2. 在线检索增强推理
- 先抽象输入问题的题型；
- 再做混合检索：题型相似 + 原题相似 + 经验相似 + 质量分；
- 支持 `zero_shot / type_only / experience_only / full` 四种模式；
- 支持检索后去冗余与多样性保留。

### 3. 实验评估与消融
- 生成每种模式下的逐题预测文件；
- 生成 `summary.csv / summary.json`；
- 内置消融：
  - full
  - zero_shot
  - type_only
  - experience_only
  - no_validation
  - no_refinement
  - no_advanced_generation

## 如何接入你的真实数据集

把真实 `math_datasets` 文件放到上一级目录 `../math_datasets` 即可。

推荐字段：
- `question`
- `answer`
- `solution`（可选）
- `dataset`（可选）
- `split`（可选）
- `subject`（可选）
- `difficulty`（可选）

如果没有 `split` 字段，也可以直接通过文件名让程序识别：
- `*_train.jsonl`
- `*_test.jsonl`
- `*_dev.jsonl`

## 快速开始

在项目目录下执行：

```bash
pip install -r requirements.txt
python scripts/build_kb.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
python scripts/run_ablation.py --config configs/default.yaml
```

如果只是想快速验证流程是否跑通：

```bash
python scripts/smoke_test.py --config configs/default.yaml
```

## 切换到真实 LLM

默认配置使用 `mock` 后端，优点是可以在无 API Key 的情况下跑通全流程。

如果你要接入真实模型，把 `configs/default.yaml` 中：

```yaml
llm:
  backend: mock
```

改成：

```yaml
llm:
  backend: openai_compatible
  model_name: your-model-name
  api_base: your-api-base
  api_key_env: OPENAI_API_KEY
```

并确保环境变量已设置。

## 与你的研究方案一一对应的模块

- 题型抽象：`src/heuristics.py` / `src/llm_backends.py`
- 经验生成：`src/heuristics.py` / `src/llm_backends.py`
- 多级验证：`src/kb_builder.py`
- 混合检索：`src/retriever.py`
- 检索后精炼：`src/retriever.py` 中 `refine=True`
- 上下文融合：`src/pipeline.py`
- 实验与消融：`src/evaluation.py` + `scripts/run_ablation.py`

