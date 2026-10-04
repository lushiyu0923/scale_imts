from __future__ import annotations

"""HumanActivity 任务构建逻辑。"""

from dataclasses import dataclass
from functools import cached_property

import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset

from ..config import DataConfig
from .raw_dataset import HumanActivityRawDataset


def activity_time_chunk(
    data: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    config: DataConfig,
) -> list[dict[str, Tensor]]:
    """把长时间活动序列切成多个历史/预测窗口样本。"""
    chunk_data: list[dict[str, Tensor]] = []
    history = config.seq_len
    pred_window = config.pred_len
    sample_id = 0

    for record_id, tt, vals, mask in data:
        t_max = int(tt.max().item())
        for start in range(0, t_max - history, 4000):
            end_x = start + history
            end_y = start + history + pred_window

            if end_x >= t_max:
                idx_x = torch.where((tt >= start) & (tt <= end_x))[0]
            else:
                idx_x = torch.where((tt >= start) & (tt < end_x))[0]

            if end_y >= t_max:
                idx_y = torch.where((tt >= end_x) & (tt <= end_y))[0]
            else:
                idx_y = torch.where((tt >= end_x) & (tt < end_y))[0]

            if len(idx_x) == 0:
                continue

            t_start = tt[idx_x][0]
            t_end = tt[idx_y][-1] + 1 if len(idx_y) > 0 else tt[idx_x][-1] + 1
            chunk_data.append(
                {
                    "sample_ID": sample_id,
                    "x_mark": (tt[idx_x] - t_start) / (t_end - t_start),
                    "y_mark": (tt[idx_y] - t_start) / (t_end - t_start),
                    "x": vals[idx_x],
                    "y": vals[idx_y],
                    "x_mask": mask[idx_x],
                    "y_mask": mask[idx_y],
                }
            )
            sample_id += 1

    return chunk_data


@dataclass
class HumanActivityChunkDataset(Dataset):
    """HumanActivity 切窗后的样本集合。"""

    samples: list[dict[str, Tensor]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.samples[index]


class HumanActivityTask:
    """独立复现 APN-main 中 HumanActivity 的任务构建方式。"""

    RANDOM_STATE = 42

    def __init__(self, config: DataConfig):
        if config.seq_len + config.pred_len > config.max_time_steps:
            raise ValueError(
                f"seq_len + pred_len must be <= {config.max_time_steps} for HumanActivity, "
                f"got {config.seq_len + config.pred_len}."
            )
        self.config = config

    @cached_property
    def records(self) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """读取并缓存原始活动记录。"""
        raw_path = self.config.raw_file_paths[0]
        dataset = HumanActivityRawDataset(self.config.dataset_root, raw_path)
        return dataset.load()

    @cached_property
    def splits(self) -> dict[str, list[dict[str, Tensor]]]:
        """按记录级切分 train / val / test，并进一步切成时间窗口。"""
        seen_data, test_data = train_test_split(
            self.records,
            train_size=0.9,
            random_state=self.RANDOM_STATE,
            shuffle=False,
        )
        train_data, val_data = train_test_split(
            seen_data,
            train_size=0.9,
            random_state=self.RANDOM_STATE,
            shuffle=False,
        )

        return {
            "train": activity_time_chunk(train_data, self.config),
            "val": activity_time_chunk(val_data, self.config),
            "test": activity_time_chunk(test_data, self.config),
        }

    def get_dataset(self, key: tuple[int, str]) -> HumanActivityChunkDataset:
        _fold, partition = key
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported partition: {partition}")
        return HumanActivityChunkDataset(self.splits[partition])

    def get_all_datasets(self, fold: int):
        return (
            self.get_dataset((fold, "train")),
            self.get_dataset((fold, "val")),
            self.get_dataset((fold, "test")),
        )
