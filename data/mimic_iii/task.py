from __future__ import annotations

"""MIMIC-III next-event forecasting task."""

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
from .raw_dataset import MIMICIIICompleteTensorDataset


@dataclass
class TaskDataset(Dataset):
    tensors: list[tuple[torch.Tensor, torch.Tensor]]
    observation_time: float
    prediction_steps: int
    idx_list: list[int]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, key: int) -> Sample:
        t, x = self.tensors[key]
        observed = t <= self.observation_time
        first_target = int(observed.sum().item())
        return Sample(
            key=self.idx_list[key],
            inputs=Inputs(
                t[:first_target],
                x[:first_target],
                t[first_target:first_target + self.prediction_steps],
            ),
            targets=x[first_target:first_target + self.prediction_steps],
            originals=(t, x),
        )


class MIMICIIITask:
    """Build the De Brouwer-style MIMIC-III forecasting splits."""

    RANDOM_STATE = 0

    def __init__(self, config: DataConfig):
        if config.seq_len + config.pred_len > config.max_time_steps:
            raise ValueError(
                "seq_len + pred_len must be <= max_time_steps for MIMIC-III, "
                f"got {config.seq_len + config.pred_len} > {config.max_time_steps}."
            )
        self.config = config
        self.prediction_steps = config.pred_len
        self.observation_time = float(config.seq_len) - 0.5

    @cached_property
    def dataset(self) -> pd.DataFrame:
        frame = MIMICIIICompleteTensorDataset(self.config.raw_csv_path).load()
        if frame.empty:
            raise ValueError("MIMIC-III tensor is empty.")

        time_max = float(frame.index.get_level_values("TIME_STAMP").max())
        if time_max <= 0:
            raise ValueError("MIMIC-III TIME_STAMP must contain a positive range.")

        normalized = frame.reset_index()
        normalized["TIME_STAMP"] = normalized["TIME_STAMP"].astype("float32") / time_max
        normalized = normalized.set_index(["UNIQUE_ID", "TIME_STAMP"])
        self.observation_time /= time_max
        return normalized

    @cached_property
    def ids(self) -> np.ndarray:
        return self.dataset.reset_index()["UNIQUE_ID"].unique()

    @cached_property
    def folds(self) -> list[dict[str, np.ndarray]]:
        folds: list[dict[str, np.ndarray]] = []
        for _ in range(self.config.num_folds):
            train_ids, test_ids = train_test_split(
                self.ids,
                test_size=self.config.test_size,
                random_state=self.RANDOM_STATE,
                shuffle=False,
            )
            train_ids, val_ids = train_test_split(
                train_ids,
                test_size=self.config.valid_size,
                random_state=self.RANDOM_STATE,
                shuffle=False,
            )
            folds.append({"train": train_ids, "val": val_ids, "test": test_ids})
        return folds

    @cached_property
    def tensors(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for record_id in self.ids:
            record_frame = self.dataset.loc[record_id]
            time = torch.tensor(record_frame.index.to_numpy(), dtype=torch.float32)
            values = torch.tensor(record_frame.to_numpy(), dtype=torch.float32)
            tensors[int(record_id)] = (time, values)
        return tensors

    def get_dataset(self, key: tuple[int, str]) -> TaskDataset:
        fold, partition = key
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported partition: {partition}")
        ids = self.folds[fold][partition]
        tensors = [self.tensors[int(record_id)] for record_id in ids]
        return TaskDataset(
            tensors=tensors,
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
            idx_list=[int(record_id) for record_id in ids],
        )

    def get_all_datasets(self, fold: int) -> Sequence[TaskDataset]:
        return (
            self.get_dataset((fold, "train")),
            self.get_dataset((fold, "val")),
            self.get_dataset((fold, "test")),
        )
