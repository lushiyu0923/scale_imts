from __future__ import annotations

"""Shared training utilities."""

import random
from pathlib import Path

import numpy as np
import torch

from scaleimts.utils.globals import logger
from scaleimts.utils.tools import EarlyStopping, ensure_dir, move_batch_to_device


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainerBase:
    def __init__(self, config):
        self.config = config
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        requested = str(requested or "cpu").lower()
        if requested.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(requested)
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are available")
            return device
        if requested.startswith("cuda"):
            logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")

    def build_optimizer(self, model: torch.nn.Module):
        name = self.config.optimizer_name.lower()
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")

    def build_scheduler(self, optimizer):
        if not self.config.use_scheduler:
            return None
        name = self.config.scheduler_name.lower()
        if name == "constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, self.config.scheduler_step_size, self.config.scheduler_gamma)
        if name == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, self.config.scheduler_gamma)
        raise ValueError(f"Unsupported scheduler: {self.config.scheduler_name}")

    def checkpoint_dir(self, task: str) -> Path:
        path = Path(self.config.save_dir) / task / self.config.dataset_name.lower()
        if task == "forecasting":
            path = path / self.config.fusion_type
        elif task == "classification":
            path = path / f"split_{self.config.split}"
        ensure_dir(path)
        return path

    @staticmethod
    def save_best(model: torch.nn.Module, path: Path) -> None:
        ensure_dir(path.parent)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_best(model: torch.nn.Module, path: Path, device: torch.device) -> None:
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
