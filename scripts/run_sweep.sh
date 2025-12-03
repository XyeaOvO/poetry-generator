#!/usr/bin/env bash

# === 第一阶段：初始化与检查 ===
# 开启严格模式：遇到变量未定义或命令报错立即退出
set -euo pipefail

CONFIG=${1:-sweep.yaml}

# 1. 检查 wandb 是否安装
if ! command -v wandb >/dev/null 2>&1; then
  echo "❌ Error: wandb CLI not found. Please install Weights & Biases first." >&2
  exit 1
fi

# 2. 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Configuration file '$CONFIG' not found."
    exit 1
fi

echo "🚀 Initializing sweep from $CONFIG..."

# 3. 创建 Sweep 并捕获输出
# 这里的技巧是把 stderr (2) 重定向到 stdout (1)，以便变量能捕获所有输出
OUTPUT=$(wandb sweep "$CONFIG" 2>&1)

# 4. 检查创建是否成功
if [ $? -ne 0 ]; then
    echo "❌ Failed to create sweep. Output:"
    echo "$OUTPUT"
    exit 1
fi

# 5. 使用 grep 和 sed 自动提取 Sweep ID
# wandb sweep 的输出通常包含一行: "Run sweep agent with: wandb agent <ID>"
SWEEP_CMD=$(echo "$OUTPUT" | grep "wandb agent" | tail -n 1)
# 去掉前面的文字，只保留 entity/project/id 部分
SWEEP_ID=${SWEEP_CMD##*wandb agent }

# 如果提取失败（为空），报错退出
if [ -z "$SWEEP_ID" ]; then
    echo "❌ Could not extract Sweep ID. Raw output:"
    echo "$OUTPUT"
    exit 1
fi

echo "✅ Sweep created successfully!"
echo "🆔 Target Sweep ID: $SWEEP_ID"
echo "---------------------------------------------------"
echo "Starting robust agent loop (Auto-restart enabled)..."
echo "Press Ctrl+C to stop."
echo "---------------------------------------------------"

# === 第二阶段：守护进程循环 ===

# ⚠️ 关键步骤：关闭 'set -e'
# 因为在这个循环中，如果 Python 脚本报错 (exit 1)，我们不希望 Shell 脚本也跟着自杀，
# 而是希望它忽略错误，继续下一次循环。
set +e

while true; do
    # 启动 agent，--count 1 确保每次只跑一个任务就退出（方便释放显存）
    wandb agent "$SWEEP_ID" --count 1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️  Agent process crashed or exited with error. Restarting in 5s..."
        sleep 5
    else
        echo "✅ Agent finished a run successfully. Starting next run in 2s..."
        sleep 2
    fi
done
