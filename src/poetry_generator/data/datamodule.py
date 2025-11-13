from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def _clean_corpus(lines: Iterable[str]) -> str:
    """Return a compact corpus string without empty lines."""
    cleaned = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned)


class _SequenceDataset(Dataset[torch.Tensor]):
    """Simple sequence dataset that yields (input, target) pairs."""

    def __init__(self, encoded_text: torch.Tensor, seq_length: int) -> None:
        if encoded_text.ndim != 1:
            raise ValueError("encoded_text must be a 1D tensor")
        self.encoded_text = encoded_text
        self.seq_length = seq_length
        if len(encoded_text) - seq_length - 1 < 0:
            raise ValueError("Corpus is too short for the requested seq_length")

    def __len__(self) -> int:
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = start + self.seq_length
        inputs = self.encoded_text[start:end]
        targets = self.encoded_text[start + 1 : end + 1]
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
        num_workers: int = 0,
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

        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

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

        corpus = Path(self.data_path).read_text(encoding="utf-8")
        corpus = _clean_corpus(corpus.splitlines())
        if not corpus:
            raise ValueError("Dataset is empty after cleaning")

        self._vocab = sorted(set(corpus))
        self._char_to_ix = {char: idx for idx, char in enumerate(self._vocab)}
        self._ix_to_char = {idx: char for char, idx in self._char_to_ix.items()}

        encoded = torch.tensor(
            [self._char_to_ix[ch] for ch in corpus],
            dtype=torch.long,
        )
        full_dataset = _SequenceDataset(encoded, self.seq_length)

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
        )
