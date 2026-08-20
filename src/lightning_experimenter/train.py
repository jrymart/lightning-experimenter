"""Train one config. This is the unit a SLURM array task executes."""

import json
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning import LightningDataModule, LightningModule

from .build import build_logger, build_trainer
from .configs import run_dir_for
from .resolve import load_class

METRICS_FILE = "metrics.json"


def train_one(cfg: dict) -> dict[str, float]:
    """Fit, validate from the best checkpoint, and write metrics.json."""
    torch.set_float32_matmul_precision(cfg["train"].get("matmul_precision", "high"))
    L.seed_everything(cfg["train"]["seed"], workers=True)

    if cfg["train"].get("require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError("train.require_cuda is set but no CUDA device is visible")

    dm = load_class(cfg["data"]["class"], LightningDataModule)(cfg)
    model = load_class(cfg["model"]["class"], LightningModule)(cfg)

    logger = build_logger(cfg)
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, (run_dir / "config.yaml").open("w"), sort_keys=False)

    trainer = build_trainer(cfg, logger, run_dir)
    trainer.fit(model, datamodule=dm)
    results = trainer.validate(model, datamodule=dm, ckpt_path="best")

    metrics = dict(results[0]) if results else {}
    if axes := cfg.get("axes"):
        logger.log_hyperparams(axes, {f"hp/{k}": v for k, v in metrics.items()})
        logger.save()

    # Written last: its presence is what marks a run complete.
    (run_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    return metrics


def train_from_file(config_path: Path) -> dict[str, float]:
    return train_one(yaml.safe_load(Path(config_path).read_text()))
