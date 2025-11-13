from __future__ import annotations

from typing import List, Sequence

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger


class LogGenerationSamplesCallback(pl.Callback):
    """Log qualitative generations to a W&B table at the end of each eval stage."""

    def __init__(
        self,
        prompts: Sequence[str],
        max_len: int = 100,
        temperature: float = 0.9,
    ) -> None:
        super().__init__()
        if not prompts:
            raise ValueError("At least one prompt is required for sampling.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.prompts = list(prompts)
        self.max_len = max_len
        self.temperature = temperature
        self._log_rows: list[list[str | int]] = []

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._log_samples(stage="val", trainer=trainer, pl_module=pl_module)

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._log_samples(stage="test", trainer=trainer, pl_module=pl_module)

    def _log_samples(
        self,
        stage: str,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.global_rank != 0:
            return
        if not hasattr(pl_module, "char_to_ix") or not hasattr(
            pl_module,
            "idx_to_char",
        ):
            pl_module.print("Sample callback skipped: missing vocabulary mapping.")
            return

        pl_module.eval()
        generated_texts: List[str] = []

        for prompt in self.prompts:
            try:
                indices = self._encode_prompt(prompt, pl_module.char_to_ix)
                if not indices:
                    raise KeyError("Prompt contains characters outside the vocabulary.")
                sample_indices = pl_module.generate(
                    start_indices=indices,
                    max_len=self.max_len,
                    temperature=self.temperature,
                )
                text = self._decode_indices(sample_indices, pl_module.idx_to_char)
            except Exception as exc:  # pragma: no cover - diagnostic path
                text = f"Generation failed: {exc}"
            generated_texts.append(text)

        pl_module.train()

        for prompt, text in zip(self.prompts, generated_texts):
            self._log_rows.append(
                [
                    int(trainer.current_epoch),
                    stage,
                    prompt,
                    text,
                ],
            )

        logger = trainer.logger
        if not isinstance(logger, WandbLogger):
            return

        columns = ["epoch", "stage", "prompt", "generated"]
        table = wandb.Table(columns=columns, data=self._log_rows)
        logger.experiment.log({f"samples/{stage}": table})

    @staticmethod
    def _encode_prompt(prompt: str, char_to_ix: dict[str, int]) -> List[int]:
        return [char_to_ix[ch] for ch in prompt if ch in char_to_ix]

    @staticmethod
    def _decode_indices(indices: Sequence[int], idx_to_char: Sequence[str]) -> str:
        vocab_size = len(idx_to_char)
        chars = [
            idx_to_char[idx] if 0 <= idx < vocab_size else ""
            for idx in indices
        ]
        return "".join(chars)
