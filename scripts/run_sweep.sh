#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-sweep.yaml}

if ! command -v wandb >/dev/null 2>&1; then
  echo "wandb CLI not found. Please install Weights & Biases first." >&2
  exit 1
fi

wandb sweep "$CONFIG"
