from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import lr_scheduler as lr_schedulers

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
        idx_to_char: List[str] | None = None,
        char_to_ix: Dict[str, int] | None = None,
        scheduler_cfg: Dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.save_hyperparameters(ignore=["idx_to_char", "char_to_ix"])

        self.model = PoetryCoreModel(
            model_type=model_type,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.idx_to_char = idx_to_char or []
        self.char_to_ix = char_to_ix or {}
        self.scheduler_cfg = scheduler_cfg or {"name": "none"}

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
        loss, batch_size = self._shared_step(batch)
        self._log_metrics("train", loss, batch_size, on_step=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, batch_size = self._shared_step(batch)
        self._log_metrics("val", loss, batch_size)
        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, batch_size = self._shared_step(batch)
        self._log_metrics("test", loss, batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler_cfg = self.scheduler_cfg or {}
        scheduler_name = (scheduler_cfg.get("name") or "none").lower()

        if scheduler_name in {"", "none"}:
            return optimizer

        scheduler = self._build_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            params=scheduler_cfg.get("params") or {},
        )
        if scheduler is None:
            return optimizer

        scheduler_dict: Dict[str, object] = {
            "scheduler": scheduler,
            "interval": scheduler_cfg.get("interval", "epoch"),
            "frequency": scheduler_cfg.get("frequency", 1),
        }

        monitor = scheduler_cfg.get("monitor")
        if not monitor and scheduler_name in {"reduce_on_plateau", "reducelronplateau"}:
            monitor = "val/loss"
        if monitor:
            scheduler_dict["monitor"] = monitor

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

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

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        inputs, targets = batch
        logits, _ = self(inputs)
        loss = self._compute_loss(logits, targets)
        return loss, inputs.size(0)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = logits.size(-1)
        loss = self.criterion(logits.view(-1, vocab_size), targets.view(-1))
        return loss

    def _perplexity(self, loss: torch.Tensor) -> torch.Tensor:
        bounded_loss = torch.clamp(loss.detach(), max=20.0)
        return torch.exp(bounded_loss)

    def _log_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        batch_size: int,
        on_step: bool = False,
    ) -> None:
        ppl = self._perplexity(loss)
        sync = self._should_sync()
        metric_prefix = f"{stage}/"

        self.log(
            f"{metric_prefix}loss",
            loss,
            prog_bar=(stage != "train" or not on_step),
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync,
        )
        self.log(
            f"{metric_prefix}ppl",
            ppl,
            prog_bar=stage != "train",
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync,
        )

    def _should_sync(self) -> bool:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        return bool(getattr(trainer, "num_devices", 1) > 1)

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: str,
        params: Dict[str, object],
    ):
        name = scheduler_name.lower()
        kwargs = dict(params)

        if name in {"step_lr", "steplr"}:
            return lr_schedulers.StepLR(optimizer, **kwargs)

        if name in {"cosine", "cosineannealing", "cosine_annealing"}:
            if "T_max" not in kwargs:
                max_epochs = getattr(self.trainer, "max_epochs", None)
                if max_epochs is None:
                    raise ValueError(
                        "Cosine scheduler requires 'T_max' or trainer.max_epochs.",
                    )
                kwargs["T_max"] = max_epochs
            return lr_schedulers.CosineAnnealingLR(optimizer, **kwargs)

        if name in {"reduce_on_plateau", "reducelronplateau"}:
            return lr_schedulers.ReduceLROnPlateau(optimizer, **kwargs)

        raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")
