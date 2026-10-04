from __future__ import annotations

"""USHCN 任务构建逻辑。

主要负责：
1. 读取原始站点级时间序列；
2. 归一化时间；
3. 划分 train / val / test fold；
4. 把整条轨迹转成预测样本。
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ..config import DataConfig
from ..sample_types import Inputs, Sample
from .raw_dataset import USHCNDeBrouwer2019RawDataset


@dataclass
class TaskDataset(Dataset):
    """把单个站点的整条轨迹切成一个预测样本。"""

    tensors: list[tuple[torch.Tensor, torch.Tensor]]
    observation_time: float
    prediction_steps: int

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, key: int) -> Sample:
        """按照 observation window / prediction window 切出输入和目标。"""
        t, x = self.tensors[key]
        observations = t <= self.observation_time
        first_target = int(observations.sum().item())
        sample_mask = slice(0, first_target)
        target_mask = slice(first_target, first_target + self.prediction_steps)
        return Sample(
            key=key,
            inputs=Inputs(t[sample_mask], x[sample_mask], t[target_mask]),
            targets=x[target_mask],
            originals=(t, x),
        )


class USHCNTask:
    """USHCN 任务逻辑，独立复现 APN-main 中的处理方式。"""

    def __init__(self, config: DataConfig):
        if config.seq_len + config.pred_len > config.max_time_steps:
            raise ValueError(
                f"seq_len + pred_len must be <= {config.max_time_steps}, "
                f"got {config.seq_len + config.pred_len}."
            )

        self.config = config
        self.prediction_steps = config.pred_len
        self.observation_time = config.seq_len - 0.5

    @cached_property
    def dataset(self) -> DataFrame:
        """读取并缓存标准化后的多变量表格。"""
        raw_dataset = USHCNDeBrouwer2019RawDataset(self.config.raw_csv_path)
        frame = raw_dataset.load()

        if self.config.normalize_time:
            frame = frame.reset_index()
            time_max = float(frame["Time"].max())
            if time_max <= 0:
                raise ValueError("USHCN Time column must contain a positive maximum value.")
            self.observation_time /= time_max
            frame["Time"] = frame["Time"] / time_max
            frame = frame.set_index(["ID", "Time"])

        return frame.dropna(axis=1, how="all").copy()

    @cached_property
    def ids(self) -> np.ndarray:
        return self.dataset.reset_index()["ID"].unique()

    @cached_property
    def folds(self) -> list[dict[str, np.ndarray]]:
        """生成固定随机种子下的多折 train/val/test 划分。"""
        saved_state = np.random.get_state()
        np.random.seed(self.config.random_seed)
        try:
            folds: list[dict[str, np.ndarray]] = []
            for _ in range(self.config.num_folds):
                train_idx, test_idx = train_test_split(
                    self.ids,
                    test_size=self.config.test_size,
                )
                train_idx, valid_idx = train_test_split(
                    train_idx,
                    test_size=self.config.valid_size,
                )
                folds.append(
                    {
                        "train": train_idx,
                        "val": valid_idx,
                        "test": test_idx,
                    }
                )
            return folds
        finally:
            np.random.set_state(saved_state)

    @cached_property
    def tensors(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """把每个站点转换成 ``(t, x)`` 张量对。"""
        tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for station_id in self.ids:
            station_frame = self.dataset.loc[station_id]
            t = torch.tensor(station_frame.index.values, dtype=torch.float32)
            x = torch.tensor(station_frame.values, dtype=torch.float32)
            tensors[int(station_id)] = (t, x)
        return tensors

    def get_dataset(self, key: tuple[int, str]) -> TaskDataset:
        """按 ``(fold, partition)`` 返回对应数据集。"""
        fold, partition = key
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported partition: {partition}")
        fold_ids = self.folds[fold][partition]
        tensors = [self.tensors[int(station_id)] for station_id in fold_ids]
        return TaskDataset(
            tensors=tensors,
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
        )

    def get_all_datasets(self, fold: int) -> Sequence[TaskDataset]:
        return (
            self.get_dataset((fold, "train")),
            self.get_dataset((fold, "val")),
            self.get_dataset((fold, "test")),
        )
