# AI 古典诗词生成器 (AI Poetry Generator)

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/PyTorch-Lightning-8A2BE2.svg)
![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

这是一个基于 RNN/LSTM 和 PyTorch Lightning 实现的古典中文诗歌生成项目，能够生成主题引导诗、藏头诗等多种类型的诗词。项目通过 Hydra 管理配置，WandB 跟踪实验，并使用 Ruff 与 Pre-commit 保证代码质量。

## ✨ 功能特性

- **主题引导**：给定任意标题或句子，模型将围绕其意境进行续写。
- **藏头诗**：按指定汉字生成整齐的藏头诗。
- **现代化工具链**：Hydra、WandB、Mamba、Ruff、Pre-commit 提升研发效率。
- **高度可复现**：清晰的脚本与配置，所有产物集中保存在 `outputs/` 目录下。

## 🚀 快速开始

### 1. 环境设置

确保已安装 [Mamba](https://github.com/mamba-org/mamba) 或 Conda，然后在项目根目录运行：

```bash
mamba env create -f environment.yml
mamba activate poetry-generator
pre-commit install
```

### 2. 数据准备

将原始诗词数据集放置于 `data/poetry.txt`（仓库已提供示例）。数据会在训练过程中自动清洗并构建词汇表。

### 3. 单次训练

所有配置均在 `conf/` 目录下定义，可通过 Hydra 覆盖。以下命令会根据配置自动在 `outputs/YYYY-MM-DD/HH-MM-SS/` 中保存 `vocab.json` 与模型检查点：

```bash
# 默认 LSTM 训练
python -m poetry_generator.pipelines.train

# 切换为 RNN 并修改 batch size
python -m poetry_generator.pipelines.train model=rnn data.batch_size=32
```

Hydra 运行目录中将包含：

- `checkpoints/best-model.ckpt`：验证损失最低的权重
- `vocab.json`：训练得到的词汇映射，可供推理脚本复用

### 4. 超参数搜索（WandB Sweeps）

`sweep.yaml` 预先配置了对模型类型、学习率、隐藏维度等的搜索策略：

```bash
# 初始化 Sweep，记录 SWEEP_ID
default_entity="<YOUR_ENTITY>"
default_project="poetry-generator"
wandb sweep sweep.yaml --entity $default_entity --project $default_project

# 运行 agent（替换为实际 Sweep 路径）
wandb agent <ENTITY/PROJECT/SWEEP_ID>
```

也可以直接运行脚本以生成 Sweep：

```bash
sh scripts/run_sweep.sh
```

### 5. 生成诗歌

训练完成后，使用 `generate.py` 加载模型与词汇表进行创作：

```bash
CKPT_PATH="outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best-model.ckpt"
VOCAB_PATH="outputs/YYYY-MM-DD/HH-MM-SS/vocab.json"

# 主题引导
python -m poetry_generator.pipelines.generate \
  --ckpt_path $CKPT_PATH \
  --vocab_path $VOCAB_PATH \
  --prompt "春江花月夜" \
  --max_len 100

# 生成藏头诗
python -m poetry_generator.pipelines.generate \
  --ckpt_path $CKPT_PATH \
  --vocab_path $VOCAB_PATH \
  --acrostic "人工智能" \
  --acrostic_line_len 48
```

`generate.py` 会自动处理字符与索引的转换，并调用 Lightning 模型内部的温度采样逻辑。

## ⚙️ 配置说明

- `conf/config.yaml`：主配置；`project_name`、`run_name` 用于 WandB 与 Hydra 命名。
- `conf/data/poetry.yaml`：数据路径、批大小、序列长度、验证集划分。
- `conf/model/*.yaml`：模型结构与学习率。可创建更多文件以扩展架构。
- `conf/trainer/default.yaml`：Lightning Trainer 参数，如 `max_epochs`、`precision`、`devices` 等。

通过 Hydra CLI 可以覆盖任意字段，例如：

```bash
python -m poetry_generator.pipelines.train \
  model=lstm \
  model.hidden_dim=512 \
  data.seq_length=64 \
  trainer.max_epochs=30
```

## 📁 项目结构

```
poetry-generator/
├── conf/
│   ├── config.yaml
│   ├── data/
│   │   └── poetry.yaml
│   ├── model/
│   │   ├── lstm.yaml
│   │   └── rnn.yaml
│   └── trainer/
│       └── default.yaml
├── data/
│   └── poetry.txt
├── scripts/
│   └── run_sweep.sh
├── src/poetry_generator/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── datamodule.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   └── lightning.py
│   └── pipelines/
│       ├── __init__.py
│       ├── generate.py
│       └── train.py
├── environment.yml
├── pyproject.toml
├── sweep.yaml
├── .pre-commit-config.yaml
├── .gitignore
└── README.md
```

## ✅ 代码质量与提交规范

1. **Ruff & Formatting**：提交前运行 `ruff check . --fix && ruff format .`。
2. **Pre-commit**：首次克隆后执行 `pre-commit install`，确保提交前自动校验。
3. **Git 工作流**：在新分支中进行开发（如 `feature/add-generator-cli`），并采用 Conventional Commits（如 `feat(model): add lstm config`）。

## 🧪 未来扩展建议

- 引入更复杂的注意力机制或 Transformer 架构。
- 增加多语种或多风格数据集。
- 结合前端界面，提供交互式诗歌体验。

祝你创作愉快！
