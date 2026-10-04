from __future__ import annotations

"""训练流程辅助工具。"""

from pathlib import Path

import numpy as np
import torch

from .globals import logger


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """把一个 batch 中的所有张量搬到指定设备。"""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


class EarlyStopping:
    """最小化 early stopping 实现，并保存验证集最优权重。"""

    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, checkpoint_path: Path) -> bool:
        """根据当前验证损失更新 early stopping 状态。"""
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            self._save_checkpoint(model, checkpoint_path)
            return True

        if score < self.best_score + self.delta:
            self.counter += 1
            logger.info("EarlyStopping counter: %s / %s", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model, checkpoint_path)
            return True

    @staticmethod
    def _save_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
        """保存当前最佳模型参数。"""
        ensure_dir(checkpoint_path.parent)
        torch.save(model.state_dict(), checkpoint_path)
