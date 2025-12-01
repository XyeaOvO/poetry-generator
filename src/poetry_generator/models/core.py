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
        embedding_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        output_dropout: float = 0.0,
        layer_norm: bool = False,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0.0 <= embedding_dropout < 1.0:
            raise ValueError("embedding_dropout must be in [0, 1)")
        if not 0.0 <= rnn_dropout < 1.0:
            raise ValueError("rnn_dropout must be in [0, 1)")
        if not 0.0 <= output_dropout < 1.0:
            raise ValueError("output_dropout must be in [0, 1)")

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
            dropout=rnn_dropout if n_layers > 1 else 0.0,
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        self.post_rnn_norm = nn.LayerNorm(hidden_dim) if layer_norm else None

        self.projection: nn.Module | None = None
        self.decoder_bias: nn.Parameter | None = None
        if tie_weights:
            if hidden_dim != embedding_dim:
                # Project hidden state to embedding dim when tying weights.
                self.projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
            self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
            self.linear = None
        else:
            self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        embedded = self.embedding_dropout(self.embedding(input_tensor))
        outputs, hidden_state = self.rnn(embedded, hidden_state)
        if self.post_rnn_norm is not None:
            outputs = self.post_rnn_norm(outputs)
        outputs = self.output_dropout(outputs)
        if self.linear is not None:
            logits = self.linear(outputs)
        else:
            if self.projection is not None:
                outputs = self.projection(outputs)
            logits = torch.nn.functional.linear(
                outputs,
                self.embedding.weight,
                bias=self.decoder_bias,
            )
        return logits, hidden_state
