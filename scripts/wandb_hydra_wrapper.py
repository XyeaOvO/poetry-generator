#!/usr/bin/env python
"""Translate W&B sweep CLI arguments into Hydra overrides."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src/poetry_generator/pipelines/train.py"


def _parse_mapping(raw_value: str, arg_name: str) -> dict[str, Any]:
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError) as exc:
        raise argparse.ArgumentTypeError(
            f"{arg_name} expects a dict literal, got: {raw_value!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"{arg_name} expects a dict literal, got: {type(parsed)!r}"
        )

    return parsed


def _format_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        normalized = "true" if value else "false"
    else:
        normalized = value
    return f"{key}={normalized}"


def _data_overrides(mapping: dict[str, Any]) -> list[str]:
    return [_format_override(f"data.{key}", value) for key, value in mapping.items()]


def _model_overrides(mapping: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    local = dict(mapping)

    model_name = local.pop("name", None)
    if model_name is not None:
        overrides.append(_format_override("model", model_name))
        overrides.append(_format_override("model.name", model_name))

    overrides.extend(
        _format_override(f"model.module.{key}", value) for key, value in local.items()
    )
    return overrides


def _build_overrides(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    if args.data is not None:
        overrides.extend(_data_overrides(_parse_mapping(args.data, "--data")))
    if args.model is not None:
        overrides.extend(_model_overrides(_parse_mapping(args.model, "--model")))
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Dict literal for data overrides")
    parser.add_argument("--model", type=str, help="Dict literal for model overrides")

    args, remaining = parser.parse_known_args()

    overrides = _build_overrides(args)
    cmd: List[str] = [sys.executable, str(TRAIN_SCRIPT), *overrides, *remaining]

    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
