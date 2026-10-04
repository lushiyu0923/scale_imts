from __future__ import annotations

"""P12 任务构建逻辑。"""

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from ..config import DataConfig
from ..sample_types import Inputs, Sample
from .raw_dataset import Physionet2012RawDataset


class Standardizer:
    """按列标准化数值特征。"""

    def __init__(self):
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        self.mean_ = frame.mean(axis=0, skipna=True)
        self.std_ = frame.std(axis=0, skipna=True).replace(0, 1.0)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        return (frame - self.mean_) / self.std_


class MinMaxTimeScaler:
    """把时间轴缩放到较稳定的数值范围。"""

    def __init__(self):
        self.xmax: float | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        time_index = frame.index.get_level_values("Time").to_numpy(dtype=np.float32)
        self.xmax = float(np.max(time_index))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.xmax is None:
            raise RuntimeError("MinMaxTimeScaler must be fit before transform.")
        normalized = frame.reset_index()
        normalized["Time"] = normalized["Time"] / (self.xmax + 1.0)
        return normalized.set_index(["RecordID", "Time"])


@dataclass
class TaskDataset(Dataset):
    """把单个病人的完整时间序列转成一个预测样本。"""

    tensors: list[tuple[torch.Tensor, torch.Tensor]]
    observation_time: float
    prediction_steps: int
    idx_list: list[int]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, key: int) -> Sample:
        t, x = self.tensors[key]
        observations = t <= self.observation_time
        first_target = int(observations.sum().item())
        sample_mask = slice(0, first_target)
        target_mask = slice(first_target, first_target + self.prediction_steps)
        return Sample(
            key=self.idx_list[key],
            inputs=Inputs(t[sample_mask], x[sample_mask], t[target_mask]),
            targets=x[target_mask],
            originals=(t, x),
        )


class P12Task:
    """独立复现 APN-main 中 P12 的任务构建逻辑。"""

    RANDOM_STATE = 0

    def __init__(self, config: DataConfig):
        if config.seq_len + config.pred_len > 96:
            raise ValueError(
                f"seq_len + pred_len must be <= 96 for P12, got {config.seq_len + config.pred_len}."
            )
        self.config = config
        self.prediction_steps = config.pred_len
        self.observation_time = float(config.seq_len)
        self.standardizer = Standardizer()
        self.time_scaler = MinMaxTimeScaler()

    @cached_property
    def dataset(self) -> pd.DataFrame:
        """读取、标准化并裁剪原始 P12 表格。"""
        raw_dataset = Physionet2012RawDataset(
            dataset_root=self.config.dataset_root,
            raw_file_paths=self.config.raw_file_paths,
        )
        frame = raw_dataset.load()
        if isinstance(frame.index, pd.MultiIndex):
            frame.index = frame.index.set_names(["RecordID", "Time"])

        self.standardizer.fit(frame)
        standardized = self.standardizer.transform(frame)
        self.time_scaler.fit(standardized)
        normalized = self.time_scaler.transform(standardized)
        if self.time_scaler.xmax is None:
            raise RuntimeError("P12 time scaler did not record xmax.")
        self.observation_time /= (self.time_scaler.xmax + 1.0)

        clipped = normalized[(-5 < normalized) & (normalized < 5)]
        return clipped.dropna(axis=1, how="all").copy()

    @cached_property
    def ids(self) -> np.ndarray:
        return self.dataset.reset_index()["RecordID"].unique()

    @cached_property
    def folds(self) -> list[dict[str, np.ndarray]]:
        """生成 train / val / test 划分。"""
        folds: list[dict[str, np.ndarray]] = []
        for _ in range(self.config.num_folds):
            train_idx, test_idx = train_test_split(
                self.ids,
                test_size=self.config.test_size,
                random_state=self.RANDOM_STATE,
                shuffle=False,
            )
            train_idx, valid_idx = train_test_split(
                train_idx,
                test_size=self.config.valid_size,
                random_state=self.RANDOM_STATE,
                shuffle=False,
            )
            folds.append(
                {
                    "train": train_idx,
                    "val": valid_idx,
                    "test": test_idx,
                }
            )
        return folds

    @cached_property
    def tensors(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """把每个病例转换成 ``(t, x)`` 张量对。"""
        tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for record_id in self.ids:
            record_frame = self.dataset.loc[record_id]
            t = torch.tensor(record_frame.index.values, dtype=torch.float32)
            x = torch.tensor(record_frame.values, dtype=torch.float32)
            tensors[int(record_id)] = (t, x)
        return tensors

    def get_dataset(self, key: tuple[int, str]) -> TaskDataset:
        fold, partition = key
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported partition: {partition}")
        fold_ids = self.folds[fold][partition]
        tensors = [self.tensors[int(record_id)] for record_id in fold_ids]
        return TaskDataset(
            tensors=tensors,
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
            idx_list=[int(record_id) for record_id in fold_ids],
        )

    def get_all_datasets(self, fold: int) -> Sequence[TaskDataset]:
        return (
            self.get_dataset((fold, "train")),
            self.get_dataset((fold, "val")),
            self.get_dataset((fold, "test")),
        )
