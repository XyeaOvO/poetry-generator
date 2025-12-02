from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from poetry_generator.models import PoetryLightningModel


def load_vocab(vocab_path: Path) -> tuple[Dict[str, int], Dict[int, str]]:
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    char_to_ix = {str(k): int(v) for k, v in data["char_to_ix"].items()}
    ix_to_char = {int(k): str(v) for k, v in data["ix_to_char"].items()}
    return {k: v for k, v in char_to_ix.items()}, ix_to_char


def encode_text(text: str, char_to_ix: Dict[str, int]) -> List[int]:
    indices = []
    for char in text:
        if char not in char_to_ix:
            raise KeyError(f"Character '{char}' not found in vocabulary")
        indices.append(char_to_ix[char])
    return indices


def decode_indices(indices: List[int], ix_to_char: Dict[int, str]) -> str:
    chars = [ix_to_char.get(idx, "") for idx in indices]
    return "".join(chars)


def remove_leading_bos(indices: List[int], bos_idx: int | None) -> List[int]:
    if bos_idx is None or not indices:
        return indices
    if indices[0] == bos_idx:
        return indices[1:]
    return indices


def truncate_sentence(text: str) -> str:
    """Return text cut at the first sentence-ending punctuation (if any)."""
    for sep in ("。", "！", "？", ".", "!", "?", "\n"):
        pos = text.find(sep)
        if pos != -1:
            return text[: pos if sep == "\n" else pos + 1]
    return text


def generate_from_prompt(
    model: PoetryLightningModel,
    char_to_ix: Dict[str, int],
    ix_to_char: Dict[int, str],
    prompt: str,
    max_len: int,
    temperature: float,
) -> str:
    bos_idx = char_to_ix.get("<bos>")
    start_indices = encode_text(prompt, char_to_ix)
    if bos_idx is not None:
        start_indices = [bos_idx] + start_indices
    generated = model.generate(
        start_indices=start_indices,
        max_len=max_len,
        temperature=temperature,
    )
    cleaned = remove_leading_bos(generated, bos_idx)
    return decode_indices(cleaned, ix_to_char)


def generate_acrostic(
    model: PoetryLightningModel,
    char_to_ix: Dict[str, int],
    ix_to_char: Dict[int, str],
    head: str,
    line_max_len: int,
    temperature: float,
) -> str:
    lines = []
    accumulated = ""
    bos_idx = char_to_ix.get("<bos>")
    for char in head:
        if char not in char_to_ix:
            raise KeyError(f"Head character '{char}' not found in vocabulary")
        prompt = accumulated + char
        start_idx = encode_text(prompt, char_to_ix)
        if bos_idx is not None:
            start_idx = [bos_idx] + start_idx
        generated = model.generate(
            start_indices=start_idx,
            max_len=len(prompt) + line_max_len,
            temperature=temperature,
        )
        cleaned = remove_leading_bos(generated, bos_idx)
        decoded = decode_indices(cleaned, ix_to_char)
        suffix = (
            decoded[len(accumulated) :] if len(decoded) > len(accumulated) else char
        )
        suffix = truncate_sentence(suffix)
        lines.append(suffix)
        accumulated += suffix
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate poetry from a trained model.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the Lightning checkpoint file",
    )
    parser.add_argument(
        "--vocab_path",
        required=True,
        help="Path to the saved vocab.json file",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Maximum length for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--acrostic_line_len",
        type=int,
        default=48,
        help="Maximum length per line when generating acrostics",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Prompt text to continue")
    group.add_argument(
        "--acrostic",
        type=str,
        help="Characters to use for an acrostic poem",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt_path)
    vocab_path = Path(args.vocab_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}")

    char_to_ix, ix_to_char = load_vocab(vocab_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoetryLightningModel.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
    )
    model.char_to_ix = char_to_ix
    model.idx_to_char = [ix_to_char[idx] for idx in range(len(ix_to_char))]
    model.to(device)
    model.eval()

    if args.prompt:
        result = generate_from_prompt(
            model,
            char_to_ix,
            ix_to_char,
            prompt=args.prompt,
            max_len=args.max_len,
            temperature=args.temperature,
        )
    else:
        result = generate_acrostic(
            model,
            char_to_ix,
            ix_to_char,
            head=args.acrostic,
            line_max_len=args.acrostic_line_len,
            temperature=args.temperature,
        )

    print(result)


if __name__ == "__main__":
    main()
