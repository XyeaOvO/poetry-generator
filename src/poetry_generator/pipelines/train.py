from __future__ import annotations

import hydra
import torch
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
import omegaconf
from omegaconf import DictConfig, OmegaConf

torch.serialization.add_safe_globals(
    [omegaconf.dictconfig.DictConfig, omegaconf.listconfig.ListConfig]
)


@hydra.main(config_path="../../../conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    abs_data_path = to_absolute_path(str(cfg.data.data_path))
    data_module = instantiate(cfg.data, data_path=abs_data_path)
    data_module.setup(stage="fit")

    vocab_mapping = data_module.vocab_mapping()
    idx_to_char = [
        vocab_mapping.ix_to_char.get(i, "") for i in range(len(vocab_mapping.vocab))
    ]
    pad_token = getattr(cfg.data, "pad_token", None)
    pad_idx = vocab_mapping.char_to_ix.get(pad_token) if pad_token else None

    model_cfg = cfg.model.module
    scheduler_cfg = OmegaConf.to_container(cfg.scheduler, resolve=True)
    assert isinstance(scheduler_cfg, dict)
    vocab_size = data_module.vocab_size
    model = instantiate(
        model_cfg,
        vocab_size=vocab_size,
        idx_to_char=idx_to_char,
        char_to_ix=vocab_mapping.char_to_ix,
        scheduler_cfg=scheduler_cfg,
        pad_idx=pad_idx,
        unk_token=getattr(cfg.data, "unk_token", None),
    )
    model = torch.compile(model, mode="max-autotune")

    wandb_logger = instantiate(cfg.logger)

    callback_cfgs = getattr(cfg, "callbacks", [])
    instantiated_callbacks = [instantiate(cb_cfg) for cb_cfg in callback_cfgs]

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_cfg, dict)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=instantiated_callbacks,
        default_root_dir=HydraConfig.get().runtime.output_dir,
        **trainer_cfg,
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    main()
