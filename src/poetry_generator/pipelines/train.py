from __future__ import annotations

import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from poetry_generator.data import PoetryDataModule
from poetry_generator.models import PoetryLightningModel


@hydra.main(config_path="../../../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42, workers=True)

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)
    data_cfg["data_path"] = to_absolute_path(
        str(data_cfg["data_path"]),
    )
    data_module = PoetryDataModule(**data_cfg)
    data_module.setup(stage="fit")

    vocab_mapping = data_module.vocab_mapping().to_dict()
    output_dir = Path.cwd()
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps(vocab_mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    model_cfg.pop("name", None)
    vocab_size = data_module.vocab_size
    model = PoetryLightningModel(vocab_size=vocab_size, **model_cfg)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.run_name,
        save_dir=str(output_dir),
        log_model=False,
    )

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_cfg, dict)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=str(output_dir),
        **trainer_cfg,
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
