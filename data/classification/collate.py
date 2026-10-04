from __future__ import annotations

"""分类任务 batch 组装。"""

from collections.abc import Callable
import zlib

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .dataset import ClassificationRecord


def _masked_stats(records: list[ClassificationRecord]) -> tuple[Tensor, Tensor, float]:
    """用当前 split 的观测值估计变量均值 / 标准差，以及时间最大值。"""

    n_var = records[0].values.shape[-1]
    sums = torch.zeros(n_var, dtype=torch.float64)
    sq_sums = torch.zeros(n_var, dtype=torch.float64)
    counts = torch.zeros(n_var, dtype=torch.float64)
    time_max = 0.0
    for record in records:
        mask = record.mask.bool()
        values = record.values.double()
        counts += mask.sum(dim=0)
        sums += torch.where(mask, values, torch.zeros_like(values)).sum(dim=0)
        sq_sums += torch.where(mask, values * values, torch.zeros_like(values)).sum(dim=0)
        if record.time.numel():
            time_max = max(time_max, float(record.time.max().item()))
    mean = torch.zeros(n_var, dtype=torch.float32)
    std = torch.ones(n_var, dtype=torch.float32)
    valid = counts > 0
    mean[valid] = (sums[valid] / counts[valid]).float()
    variance = torch.clamp(sq_sums / counts.clamp_min(1.0) - (sums / counts.clamp_min(1.0)) ** 2, min=0.0)
    std[valid] = torch.sqrt(variance[valid]).float().clamp_min(1e-8)
    return mean, std, max(time_max, 1e-8)


class ClassificationNormalizer:
    """训练集 + 验证集上的变量标准化和时间缩放。"""

    def __init__(self, records: list[ClassificationRecord]):
        if not records:
            raise ValueError("Cannot fit a classification normalizer on an empty record list.")
        self.mean, self.std, self.time_max = _masked_stats(records)

    def transform(self, record: ClassificationRecord) -> ClassificationRecord:
        values = (record.values - self.mean) / self.std
        values = torch.where(record.mask.bool(), values, torch.zeros_like(values))
        time = record.time / self.time_max
        return ClassificationRecord(
            sample_id=record.sample_id,
            values=values.float(),
            time=time.float(),
            mask=record.mask.float(),
            label=record.label,
        )


def build_classification_collate(
    seq_len_max_irr: int,
) -> Callable[[list[ClassificationRecord]], dict[str, Tensor]]:
    def _collate(batch: list[ClassificationRecord]) -> dict[str, Tensor]:
        xs: list[Tensor] = []
        times: list[Tensor] = []
        masks: list[Tensor] = []
        labels: list[int] = []
        for record in batch:
            xs.append(record.values)
            times.append(record.time.view(-1))
            masks.append(record.mask)
            labels.append(record.label)

        n_var = xs[0].shape[-1]
        xs.append(torch.zeros(seq_len_max_irr, n_var))
        times.append(torch.zeros(seq_len_max_irr))
        masks.append(torch.zeros(seq_len_max_irr, n_var))

        values = pad_sequence(xs, batch_first=True)[:-1]
        time = pad_sequence(times, batch_first=True)[:-1].unsqueeze(-1)
        mask = pad_sequence(masks, batch_first=True)[:-1]
        dummy_y_mark = torch.ones(len(batch), 1, 1, dtype=time.dtype)
        if time.numel():
            valid_time = time.squeeze(-1)
            valid = mask.any(dim=-1)
            filled = torch.where(valid, valid_time, torch.zeros_like(valid_time))
            last_time = filled.max(dim=1, keepdim=True).values.unsqueeze(-1)
            dummy_y_mark = torch.clamp(last_time, min=0.0)

        return {
            "x": values.float(),
            "x_mark": time.float(),
            "x_mask": mask.float(),
            "sample_ID": torch.tensor(
                [
                    zlib.crc32(str(record.sample_id or index).encode("utf-8"))
                    for index, record in enumerate(batch)
                ],
                dtype=torch.float32,
            ),
            "y_mark": dummy_y_mark.float(),
            "y": torch.tensor(labels, dtype=torch.long),
            "y_mask": torch.ones(len(batch), 1, dtype=torch.float32),
        }

    return _collate
