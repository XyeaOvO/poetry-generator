from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class PoetryCoreModel(nn.Module):
    """Plain PyTorch model backing the LightningModule."""

    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.model_type = model_type.lower()
        if self.model_type not in {"rnn", "lstm"}:
            raise ValueError("model_type must be either 'rnn' or 'lstm'")

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        rnn_cls = nn.LSTM if self.model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        embedded = self.embedding(input_tensor)
        outputs, hidden_state = self.rnn(embedded, hidden_state)
        logits = self.linear(outputs)
        return logits, hidden_state
