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
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        pad_idx: int | None = None,
        embedding_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        output_dropout: float = 0.0,
        layer_norm: bool = False,
        tie_weights: bool = False,
        idx_to_char: List[str] | None = None,
        char_to_ix: Dict[str, int] | None = None,
        scheduler_cfg: Dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if pad_idx is not None and pad_idx < 0:
            raise ValueError("pad_idx must be non-negative when provided")
        self.save_hyperparameters(ignore=["idx_to_char", "char_to_ix"])

        self.model = PoetryCoreModel(
            model_type=model_type,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            embedding_dropout=embedding_dropout,
            rnn_dropout=rnn_dropout,
            output_dropout=output_dropout,
            layer_norm=layer_norm,
            tie_weights=tie_weights,
        )
        ce_kwargs: dict[str, object] = {"label_smoothing": label_smoothing}
        if pad_idx is not None:
            ce_kwargs["ignore_index"] = pad_idx
        self.criterion = nn.CrossEntropyLoss(**ce_kwargs)
        self.idx_to_char = idx_to_char or []
        self.char_to_ix = char_to_ix or {}
        self.scheduler_cfg = scheduler_cfg or {"name": "none"}
        self.pad_idx = pad_idx

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
        hiddens: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
    ):
        loss, batch_size, new_hidden = self._forward_with_loss(batch, hiddens)
        self._log_metrics("train", loss, batch_size, on_step=True)
        new_hidden = self._detach_hidden(new_hidden)
        trainer = getattr(self, "trainer", None)
        if trainer and getattr(trainer, "truncated_bptt_steps", 0):
            return loss, new_hidden
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, batch_size, _ = self._forward_with_loss(batch)
        self._log_metrics("val", loss, batch_size)
        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, batch_size, _ = self._forward_with_loss(batch)
        self._log_metrics("test", loss, batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
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
        top_k: int | None = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_newlines: int | None = 6,
        min_tokens_between_newlines: int = 4,
        eos_idx: int | None = None,
    ) -> List[int]:
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not start_indices:
            raise ValueError("start_indices cannot be empty")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if max_newlines is not None and max_newlines <= 0:
            raise ValueError("max_newlines must be positive or None")
        if min_tokens_between_newlines < 0:
            raise ValueError("min_tokens_between_newlines must be non-negative")

        device = self.device
        generated = list(start_indices)
        self.eval()
        self.model.eval()

        newline_idx = self.char_to_ix.get("\n") if hasattr(self, "char_to_ix") else None
        tokens_since_newline = self._tokens_since_newline(generated, newline_idx)
        newline_count = generated.count(newline_idx) if newline_idx is not None else 0
        eos_index = eos_idx
        if eos_index is None and hasattr(self, "char_to_ix"):
            eos_index = self.char_to_ix.get("<eos>")

        with torch.no_grad():
            start_tensor = torch.tensor(
                generated,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            _, hidden = self.model(start_tensor, None)

            current_input = torch.empty((1, 1), dtype=torch.long, device=device)
            last_token = generated[-1]

            while len(generated) < max_len:
                current_input.fill_(last_token)
                logits, hidden = self.model(current_input, hidden)
                logits = logits[:, -1, :].squeeze(0) / temperature
                logits = self._apply_repetition_penalty(
                    logits,
                    generated,
                    repetition_penalty,
                )
                logits = self._mask_newlines(
                    logits,
                    newline_idx,
                    newline_count,
                    tokens_since_newline,
                    max_newlines,
                    min_tokens_between_newlines,
                )
                logits = self._filter_logits(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(logits, dim=-1)
                probs = probs / probs.sum()
                next_token = torch.multinomial(probs, num_samples=1)
                last_token = next_token.item()
                generated.append(last_token)
                if eos_index is not None and last_token == eos_index:
                    break
                tokens_since_newline += 1
                if newline_idx is not None and last_token == newline_idx:
                    newline_count += 1
                    tokens_since_newline = 0

        return generated[:max_len]

    def _forward_with_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[
        torch.Tensor, int, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
    ]:
        inputs, targets = batch
        logits, hidden = self(inputs, hidden_state)
        loss = self._compute_loss(logits, targets)
        return loss, inputs.size(0), hidden

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

    @staticmethod
    def _tokens_since_newline(
        generated: Sequence[int],
        newline_idx: int | None,
    ) -> int:
        if newline_idx is None:
            return len(generated)
        for offset, token in enumerate(reversed(generated), 1):
            if token == newline_idx:
                return offset - 1
        return len(generated)

    @staticmethod
    def _detach_hidden(
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        if hidden is None:
            return None
        if isinstance(hidden, tuple):
            return tuple(h.detach() for h in hidden)
        return hidden.detach()

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated: Sequence[int],
        repetition_penalty: float,
    ) -> torch.Tensor:
        if repetition_penalty == 1.0 or not generated:
            return logits
        vocab_size = logits.size(-1)
        token_tensor = torch.tensor(generated, device=logits.device)
        counts = torch.bincount(token_tensor, minlength=vocab_size).bool()
        penalized = logits.clone()
        penalized[counts & (penalized < 0)] *= repetition_penalty
        penalized[counts & (penalized >= 0)] /= repetition_penalty
        return penalized

    @staticmethod
    def _mask_newlines(
        logits: torch.Tensor,
        newline_idx: int | None,
        newline_count: int,
        tokens_since_newline: int,
        max_newlines: int | None,
        min_tokens_between_newlines: int,
    ) -> torch.Tensor:
        if newline_idx is None:
            return logits
        masked = logits.clone()
        if max_newlines is not None and newline_count >= max_newlines:
            masked[newline_idx] = -torch.inf
        elif tokens_since_newline < min_tokens_between_newlines:
            masked[newline_idx] = -torch.inf
        return masked

    @staticmethod
    def _filter_logits(
        logits: torch.Tensor,
        top_k: int | None = None,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        filtered = logits.clone()
        if top_k is not None and top_k > 0 and top_k < filtered.size(-1):
            topk = torch.topk(filtered, top_k)
            min_topk = topk.values[..., -1, None]
            filtered = torch.where(
                filtered < min_topk,
                torch.tensor(-torch.inf, device=logits.device),
                filtered,
            )

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > top_p
            cutoff[..., 0] = False
            sorted_logits = torch.where(
                cutoff, torch.tensor(-torch.inf, device=logits.device), sorted_logits
            )
            # Re-map back to original ordering.
            filtered = torch.full_like(filtered, -torch.inf)
            filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return filtered

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
            return lr_schedulers.CosineAnnealingLR(optimizer, **kwargs)

        if name in {"reduce_on_plateau", "reducelronplateau"}:
            return lr_schedulers.ReduceLROnPlateau(optimizer, **kwargs)

        raise ValueError(f"Unsupported scheduler '{scheduler_name}'.")
