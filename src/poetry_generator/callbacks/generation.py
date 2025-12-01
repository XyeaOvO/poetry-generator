from __future__ import annotations

from typing import List, Sequence

import traceback

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger


class LogGenerationSamplesCallback(pl.Callback):
    """Log qualitative generations to a W&B table at the end of each eval stage."""

    def __init__(
        self,
        prompts: Sequence[str] | None = None,
        acrostic_heads: Sequence[str] | None = None,
        acrostic_line_len: int = 48,
        long_prompts: Sequence[str] | None = None,
        long_max_len: int = 512,
        long_min_lines: int = 20,
        max_len: int = 100,
        temperature: float = 0.9,
    ) -> None:
        super().__init__()
        if not (prompts or acrostic_heads or long_prompts):
            raise ValueError(
                "At least one of prompts/acrostic_heads/long_prompts is required."
            )
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if acrostic_heads is not None and acrostic_line_len <= 0:
            raise ValueError("acrostic_line_len must be positive.")
        if long_prompts is not None and long_max_len <= 0:
            raise ValueError("long_max_len must be positive.")
        if long_prompts is not None and long_min_lines <= 0:
            raise ValueError("long_min_lines must be positive.")

        self.prompts = list(prompts) if prompts else []
        self.acrostic_heads = list(acrostic_heads) if acrostic_heads else []
        self.acrostic_line_len = acrostic_line_len
        self.long_prompts = list(long_prompts) if long_prompts else []
        self.long_max_len = long_max_len
        self.long_min_lines = long_min_lines
        self.max_len = max_len
        self.temperature = temperature
        self._stage_tables: dict[str, wandb.Table] = {}

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
        stage_rows: list[list[str | int]] = []

        # Prompt continuation
        for prompt in self.prompts:
            text = self._safe_generate(
                pl_module,
                prompt,
                max_len=self.max_len,
                temperature=self.temperature,
            )
            stage_rows.append(
                [
                    int(trainer.current_epoch),
                    stage,
                    "prompt",
                    prompt,
                    text,
                ],
            )

        # Acrostic poems
        for head in self.acrostic_heads:
            try:
                lines: list[str] = []
                accumulated = ""
                for ch in head:
                    line = self._generate_acrostic_line(
                        pl_module=pl_module,
                        accumulated=accumulated,
                        head_char=ch,
                    )
                    lines.append(line)
                    accumulated += line
                text = "\n".join(lines)
            except Exception:  # pragma: no cover - unexpected failure should raise
                tb = traceback.format_exc()
                pl_module.print(
                    "Sample callback failed while generating acrostic"
                    f" '{head}':\n{tb}",
                )
                raise

            stage_rows.append(
                [
                    int(trainer.current_epoch),
                    stage,
                    "acrostic",
                    head,
                    text,
                ],
            )

        # Long-form poems
        for prompt in self.long_prompts:
            text = self._safe_generate(
                pl_module,
                prompt,
                max_len=self.long_max_len,
                temperature=self.temperature,
            )
            lines = text.count("\n") + 1

            stage_rows.append(
                [
                    int(trainer.current_epoch),
                    stage,
                    "long",
                    prompt,
                    text,
                ],
            )

        pl_module.train()  # type: ignore[call-arg]

        logger = trainer.logger
        if not isinstance(logger, WandbLogger):
            return

        columns = ["epoch", "stage", "kind", "prompt", "generated"]
        table = self._stage_tables.get(stage)
        if table is None:
            table = wandb.Table(columns=columns, log_mode="MUTABLE")
            self._stage_tables[stage] = table

        for row in stage_rows:
            table.add_data(*row)
        logger.experiment.log({f"samples/{stage}": table})

    @staticmethod
    def _encode_prompt(prompt: str, char_to_ix: dict[str, int]) -> List[int]:
        missing_chars = [ch for ch in prompt if ch not in char_to_ix]
        if missing_chars:
            # 使用集合保持信息紧凑，同时保留出现顺序方便排查
            seen = []
            for ch in missing_chars:
                if ch not in seen:
                    seen.append(ch)
            formatted = "', '".join(seen)
            raise KeyError(
                f"Prompt contains characters outside the vocabulary: '{formatted}'"
            )
        return [char_to_ix[ch] for ch in prompt]

    @staticmethod
    def _decode_indices(indices: Sequence[int], idx_to_char: Sequence[str]) -> str:
        vocab_size = len(idx_to_char)
        chars = [idx_to_char[idx] if 0 <= idx < vocab_size else "" for idx in indices]
        return "".join(chars)

    @staticmethod
    def _truncate_sentence(text: str) -> str:
        """Cut at the first sentence-ending punctuation if present."""
        for sep in ("。", "！", "？", ".", "!", "?"):
            pos = text.find(sep)
            if pos != -1:
                return text[: pos + 1]
        return text

    def _generate_acrostic_line(
        self,
        pl_module: pl.LightningModule,
        accumulated: str,
        head_char: str,
    ) -> str:
        prompt = accumulated + head_char
        raw = self._safe_generate(
            pl_module,
            prompt,
            max_len=len(prompt) + self.acrostic_line_len,
            temperature=self.temperature,
        )
        suffix = raw[len(accumulated) :] if len(raw) > len(accumulated) else head_char
        line = self._truncate_sentence(suffix)
        return line

    def _safe_generate(
        self,
        pl_module: pl.LightningModule,
        prompt: str,
        max_len: int,
        temperature: float,
        eos_idx: int | None = None,
    ) -> str:
        try:
            indices = self._encode_prompt(prompt, pl_module.char_to_ix)
        except KeyError as exc:
            return f"Generation skipped: {exc}"

        try:
            sample_indices = pl_module.generate(
                start_indices=indices,
                max_len=max_len,
                temperature=temperature,
                eos_idx=eos_idx,
            )
            return self._decode_indices(sample_indices, pl_module.idx_to_char)
        except Exception:  # pragma: no cover - unexpected failure should raise
            tb = traceback.format_exc()
            pl_module.print(
                "Sample callback failed while generating text for prompt"
                f" '{prompt}':\n{tb}",
            )
            raise
