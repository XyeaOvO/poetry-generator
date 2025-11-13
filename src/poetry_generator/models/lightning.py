from __future__ import annotations

from typing import List, Sequence

import pytorch_lightning as pl
import torch
from torch import nn

from .core import PoetryCoreModel


class PoetryLightningModel(pl.LightningModule):
    """LightningModule that wraps the core PyTorch model."""

    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        learning_rate: float,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.save_hyperparameters()

        self.model = PoetryCoreModel(
            model_type=model_type,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        return self.model(inputs, hidden_state)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        logits, _ = self(inputs)
        loss = self._compute_loss(logits, targets)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=inputs.size(0),
        )
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        logits, _ = self(inputs)
        loss = self._compute_loss(logits, targets)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=inputs.size(0),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def generate(
        self,
        start_indices: Sequence[int],
        max_len: int = 100,
        temperature: float = 1.0,
    ) -> List[int]:
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not start_indices:
            raise ValueError("start_indices cannot be empty")

        device = self.device
        generated = list(start_indices)
        self.eval()
        self.model.eval()

        with torch.no_grad():
            start_tensor = torch.tensor(
                generated,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            _, hidden = self.model(start_tensor, None)

            while len(generated) < max_len:
                current_input = torch.tensor(
                    [[generated[-1]]],
                    dtype=torch.long,
                    device=device,
                )
                logits, hidden = self.model(current_input, hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

        return generated[:max_len]

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = logits.size(-1)
        loss = self.criterion(logits.view(-1, vocab_size), targets.view(-1))
        return loss
