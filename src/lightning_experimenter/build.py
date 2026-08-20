"""Construct the Lightning objects a run needs, from a config dict."""

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def build_logger(cfg: dict) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=cfg["train"].get("log_dir", "tb_logs"),
        name=cfg["name_tag"],
        version=cfg["version_tag"],
    )


def build_schedule(train_cfg: dict) -> dict:
    """Trainer kwargs for step-based or epoch-based training.

    The four keys move together: `val_check_interval` counts steps only when
    `check_val_every_n_epoch` is None, and `max_epochs` must be pinned to -1 in
    step mode or Lightning substitutes its default of 1000 and caps the run.
    """
    max_steps = train_cfg.get("max_steps")
    max_epochs = train_cfg.get("max_epochs")
    if (max_steps is None) == (max_epochs is None):
        raise ValueError("set exactly one of train.max_steps / train.max_epochs")

    if max_steps is not None:
        return {
            "max_steps": max_steps,
            "max_epochs": -1,
            "check_val_every_n_epoch": None,
            "val_check_interval": train_cfg["val_every_n_steps"],
        }
    else:
        return {
            "max_steps": -1,
            "max_epochs": max_epochs,
            "check_val_every_n_epoch": train_cfg.get("val_every_n_epochs", 1),
            "val_check_interval": 1.0,
        }


def build_trainer(cfg: dict, logger, run_dir: Path) -> L.Trainer:
    train_cfg = cfg["train"]
    return L.Trainer(
        logger=logger,
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        callbacks=[
            ModelCheckpoint(
                monitor=train_cfg["monitor"],
                mode=train_cfg.get("monitor_mode", "min"),
                save_top_k=1,
                dirpath=run_dir,
                filename="best",
            ),
        ],
        **build_schedule(train_cfg),
    )
