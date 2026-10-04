from __future__ import annotations

"""官方 IMTS 分类样本的读取与标准化。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .protocol import (
    DATASET_SPECS,
    candidate_roots,
    missing_file_message,
    resolve_existing_file,
    split_file_candidates,
)


@dataclass
class ClassificationRecord:
    """单个分类样本。"""

    sample_id: str
    values: Tensor
    time: Tensor
    mask: Tensor
    label: int


def _as_1d_time(time_values: np.ndarray) -> np.ndarray:
    time = np.asarray(time_values, dtype=np.float32).reshape(-1)
    return time


def _trim_trailing_padding(time: np.ndarray, values: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """去掉时间戳后面的零填充，保持和 ISTS-PLM 一致。"""

    if time.size <= 1:
        if mask is None:
            mask = (np.abs(values) > 0).astype(np.float32)
        return time, values, mask

    zero_after_first = np.where(time[1:] == 0)[0]
    if zero_after_first.size:
        end = int(zero_after_first[0] + 1)
        time = time[:end]
        values = values[:end]
        if mask is not None:
            mask = mask[:end]
    if mask is None:
        mask = (np.abs(values) > 0).astype(np.float32)
    else:
        mask = mask[: time.shape[0]]
    return time, values, mask


def _record_from_ptdict(item, label: int, dataset_name: str) -> ClassificationRecord:
    if isinstance(item, dict):
        sample_id = str(item.get("id", ""))
        values = np.asarray(item["arr"], dtype=np.float32)
        time = _as_1d_time(item["time"])
        length = int(item.get("length", time.shape[0]))
        length = max(1, min(length, values.shape[0], time.shape[0]))
        values = values[:length]
        time = time[:length]
        if "mask" in item:
            mask = np.asarray(item["mask"], dtype=np.float32)[:length]
        else:
            mask = (np.abs(values) > 0).astype(np.float32)
        time, values, mask = _trim_trailing_padding(time, values, mask)
    else:
        values = np.asarray(item, dtype=np.float32)
        time = np.linspace(0.0, float(values.shape[0]), values.shape[0], dtype=np.float32)
        if dataset_name == "PAM":
            time = time / 60.0
        mask = (np.abs(values) > 0).astype(np.float32)
        sample_id = ""

    if values.ndim != 2:
        raise ValueError(f"Expected 2D values, got shape {values.shape}")
    return ClassificationRecord(
        sample_id=sample_id,
        values=torch.from_numpy(np.ascontiguousarray(values)),
        time=torch.from_numpy(np.ascontiguousarray(time)).view(-1, 1),
        mask=torch.from_numpy(np.ascontiguousarray(mask)),
        label=int(label),
    )


def _record_from_mimic(item, label: int) -> ClassificationRecord:
    sample_id, timestamps, values, mask, length = item
    length = int(length)
    values = np.asarray(values[:length], dtype=np.float32)
    timestamps = np.asarray(timestamps[:length], dtype=np.float32).reshape(-1)
    mask = np.asarray(mask[:length], dtype=np.float32)
    return ClassificationRecord(
        sample_id=str(sample_id),
        values=torch.from_numpy(np.ascontiguousarray(values)),
        time=torch.from_numpy(np.ascontiguousarray(timestamps)).view(-1, 1),
        mask=torch.from_numpy(np.ascontiguousarray(mask)),
        label=int(label),
    )


def _load_split_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed = np.load(path, allow_pickle=True)
    if len(packed) != 3:
        raise ValueError(f"Expected (train, val, test) indices in {path}, got {len(packed)} arrays.")
    return packed[0], packed[1], packed[2]


def load_raindrop_records(dataset_name: str, split: int, flag: str, data_root: str | Path | None = None) -> tuple[list[ClassificationRecord], int]:
    """读取 Raindrop/ISTS-PLM 风格的 P12/P19/PAM 分类样本。"""

    name = dataset_name.upper()
    spec = DATASET_SPECS[name]
    roots = candidate_roots(name, data_root)
    dict_path = resolve_existing_file([root / rel for root in roots for rel in spec.dict_names])
    outcome_path = resolve_existing_file([root / rel for root in roots for rel in spec.outcome_names])
    split_path = resolve_existing_file(split_file_candidates(name, split, data_root))
    if dict_path is None or outcome_path is None or split_path is None:
        raise FileNotFoundError(missing_file_message(name, data_root))

    records = np.load(dict_path, allow_pickle=True)
    outcomes = np.load(outcome_path, allow_pickle=True)
    labels = np.asarray(outcomes).reshape(len(outcomes), -1)[:, -1].astype(np.int64)
    if len(records) != len(labels):
        raise ValueError(
            f"{name} feature/label size mismatch: {len(records)} records vs {len(labels)} labels."
        )

    idx_train, idx_val, idx_test = _load_split_arrays(split_path)
    split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
    if flag not in split_map:
        raise ValueError(f"Unsupported split flag: {flag}")

    selected = []
    for index in np.asarray(split_map[flag]).reshape(-1):
        selected.append(_record_from_ptdict(records[int(index)], int(labels[int(index)]), name))
    n_variables = int(selected[0].values.shape[-1]) if selected else spec.n_variables
    return selected, n_variables


def load_mimic_records(flag: str, data_root: str | Path | None = None) -> tuple[list[ClassificationRecord], int]:
    """读取已经处理好的 MIMIC-III in-hospital mortality 分类样本。"""

    root = candidate_roots("MIMIC_III", data_root)[0]
    x_path = root / f"mimic3_{flag}_x.npy"
    y_path = root / f"mimic3_{flag}_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"MIMIC-III classification files were not found under {root}. "
            "Expected mimic3_{train,val,test}_x.npy and corresponding y files."
        )
    x_records = np.load(x_path, allow_pickle=True)
    y_records = np.load(y_path, allow_pickle=True).reshape(-1)
    selected = [
        _record_from_mimic(item, int(label))
        for item, label in zip(x_records, y_records)
    ]
    n_variables = int(selected[0].values.shape[-1]) if selected else DATASET_SPECS["MIMIC_III"].n_variables
    return selected, n_variables


class ClassificationWrapperDataset(Dataset):
    """把官方分类样本转成 ScaleIMTS 可消费的历史序列。"""

    def __init__(self, dataset_name: str, split: int, flag: str, data_root: str | Path | None = None):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {flag}")
        self.dataset_name = dataset_name.upper()
        self.split = int(split)
        self.flag = flag
        self.data_root = data_root
        if self.dataset_name == "MIMIC_III":
            self.records, self.n_variables = load_mimic_records(flag, data_root)
        else:
            self.records, self.n_variables = load_raindrop_records(self.dataset_name, self.split, flag, data_root)
        self.n_classes = DATASET_SPECS[self.dataset_name].n_classes
        self.max_seq_len = max((item.values.shape[0] for item in self.records), default=1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ClassificationRecord:
        return self.records[index]
