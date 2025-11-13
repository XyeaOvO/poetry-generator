from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl


class SaveVocabCallback(pl.Callback):
    """Persist the vocabulary mapping to disk at the start of training."""

    def __init__(self, output_path: str = "vocab.json") -> None:
        super().__init__()
        self.output_path = Path(output_path)

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:  # pragma: no cover
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "vocab_mapping"):
            pl_module.print(
                "SaveVocabCallback skipped: datamodule has no vocab_mapping().",
            )
            return

        try:
            vocab_dict = datamodule.vocab_mapping().to_dict()
        except Exception as exc:  # pragma: no cover
            pl_module.print(f"SaveVocabCallback failed to read vocab: {exc}")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(vocab_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pl_module.print(f"Vocabulary saved to {self.output_path}")
