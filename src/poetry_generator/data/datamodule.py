from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def _split_poems(lines: Iterable[str]) -> list[str]:
    """Split raw lines into poems separated by blank lines."""
    poems: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            current.append(stripped)
            continue
        if current:
            poems.append("\n".join(current))
            current = []
    if current:
        poems.append("\n".join(current))
    return poems


class _PoemWindowDataset(Dataset[torch.Tensor]):
    """Dataset that samples fixed-length windows within each poem."""

    def __init__(self, poems: List[torch.Tensor], seq_length: int) -> None:
        self.poems = poems
        self.seq_length = seq_length
        self._indices: list[tuple[int, int]] = []

        for poem_idx, poem in enumerate(self.poems):
            if poem.ndim != 1:
                raise ValueError("Each poem tensor must be 1D")
            max_start = poem.size(0) - seq_length - 1
            if max_start < 0:
                continue
            self._indices.extend((poem_idx, start) for start in range(max_start + 1))

        if not self._indices:
            raise ValueError("Corpus is too short for the requested seq_length")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        poem_idx, start = self._indices[idx]
        poem = self.poems[poem_idx]
        end = start + self.seq_length
        inputs = poem[start:end]
        targets = poem[start + 1 : end + 1]
        return inputs, targets


@dataclass
class VocabMapping:
    vocab: list[str]
    char_to_ix: Dict[str, int]
    ix_to_char: Dict[int, str]

    def to_dict(self) -> Dict[str, Dict[str, int] | list[str]]:
        return {
            "vocab": self.vocab,
            "char_to_ix": self.char_to_ix,
            "ix_to_char": {str(k): v for k, v in self.ix_to_char.items()},
        }


class PoetryDataModule(pl.LightningDataModule):
    """LightningDataModule responsible for preparing poetry datasets."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        seq_length: int,
        val_split: float,
        test_split: float,
        num_workers: int = 20,
        max_vocab_size: Optional[int] = 4000,
        min_char_freq: int = 1,
        unk_token: str = "�",
        separator_token: str = "|",
        append_separator: bool = True,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seq_length <= 1:
            raise ValueError("seq_length must be greater than 1")
        if not 0 < val_split < 1:
            raise ValueError("val_split must be in the range (0, 1)")
        if not 0 < test_split < 1:
            raise ValueError("test_split must be in the range (0, 1)")
        if val_split + test_split >= 1:
            raise ValueError("The sum of val_split and test_split must be < 1")
        if max_vocab_size is not None and max_vocab_size <= 0:
            raise ValueError("max_vocab_size must be positive or None")
        if min_char_freq <= 0:
            raise ValueError("min_char_freq must be positive")
        if not unk_token:
            raise ValueError("unk_token cannot be empty")
        if len(separator_token) != 1:
            raise ValueError("separator_token must be a single character")
        if len(unk_token) != 1:
            raise ValueError("unk_token must be a single character")

        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.max_vocab_size = max_vocab_size
        self.min_char_freq = min_char_freq
        self.unk_token = unk_token
        self.separator_token = separator_token
        self.append_separator = append_separator

        self._vocab: list[str] = []
        self._char_to_ix: Dict[str, int] = {}
        self._ix_to_char: Dict[int, str] = {}
        self._train_dataset: Optional[Dataset[tuple[torch.Tensor, torch.Tensor]]] = None
        self._val_dataset: Optional[Dataset[tuple[torch.Tensor, torch.Tensor]]] = None
        self._test_dataset: Optional[Dataset[tuple[torch.Tensor, torch.Tensor]]] = None

    def prepare_data(self) -> None:  # pragma: no cover - filesystem check
        path = Path(self.data_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_dataset is not None and self._val_dataset is not None:
            return

        raw_text = Path(self.data_path).read_text(encoding="utf-8")
        poems = _split_poems(raw_text.splitlines())
        if not poems:
            raise ValueError("Dataset is empty after cleaning")

        self._vocab = self._build_vocab(poems)
        self._char_to_ix = {char: idx for idx, char in enumerate(self._vocab)}
        self._ix_to_char = {idx: char for char, idx in self._char_to_ix.items()}

        unk_idx = self._char_to_ix[self.unk_token]
        sep_idx = self._char_to_ix[self.separator_token]

        encoded_poems: list[torch.Tensor] = []
        for poem in poems:
            encoded = [self._char_to_ix.get(ch, unk_idx) for ch in poem]
            if self.append_separator:
                encoded.append(sep_idx)
            encoded_poems.append(torch.tensor(encoded, dtype=torch.long))

        full_dataset = _PoemWindowDataset(encoded_poems, self.seq_length)

        dataset_len = len(full_dataset)
        val_size = int(dataset_len * self.val_split)
        test_size = int(dataset_len * self.test_split)
        if val_size == 0:
            val_size = 1
        if test_size == 0:
            test_size = 1
        train_size = dataset_len - val_size - test_size
        if train_size <= 0:
            raise ValueError("Validation/Test split too large for dataset size")

        splits = random_split(
            full_dataset,
            lengths=[train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        self._train_dataset, self._val_dataset, self._test_dataset = splits

    @property
    def vocab(self) -> list[str]:
        if not self._vocab:
            raise RuntimeError("Vocabulary is not initialized. Call setup() first.")
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def vocab_mapping(self) -> VocabMapping:
        return VocabMapping(self.vocab, self._char_to_ix, self._ix_to_char)

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError(
                "DataModule must be set up before requesting dataloaders",
            )
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError(
                "DataModule must be set up before requesting dataloaders",
            )
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_dataset is None:
            raise RuntimeError(
                "DataModule must be set up before requesting dataloaders",
            )
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def _build_vocab(self, poems: list[str]) -> list[str]:
        counter: Counter[str] = Counter()
        for poem in poems:
            counter.update(poem)

        filtered = {
            ch: freq for ch, freq in counter.items() if freq >= self.min_char_freq
        }
        # Sort by frequency (desc) then by character to keep determinism.
        sorted_items = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))
        if self.max_vocab_size is not None:
            sorted_items = sorted_items[: self.max_vocab_size]

        vocab = [self.separator_token, self.unk_token]
        vocab.extend(
            ch
            for ch, _ in sorted_items
            if ch not in {self.separator_token, self.unk_token}
        )
        return vocab
