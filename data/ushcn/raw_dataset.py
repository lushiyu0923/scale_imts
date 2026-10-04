from __future__ import annotations

"""USHCN 原始稀疏 CSV 的读取与重排。"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame


REQUIRED_COLUMNS = [
    "ID",
    "Time",
    "Value_0",
    "Value_1",
    "Value_2",
    "Value_3",
    "Value_4",
    "Mask_0",
    "Mask_1",
    "Mask_2",
    "Mask_3",
    "Mask_4",
]


@dataclass
class USHCNDeBrouwer2019RawDataset:
    """读取 APN-main 使用的 USHCN 稀疏 CSV。"""

    csv_path: Path

    def load(self) -> DataFrame:
        """读取 CSV，并转换成多变量时间表。"""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"USHCN sparse CSV not found: {self.csv_path}. "
                "Place small_chunked_sporadic.csv here or let the demo main create one."
            )

        dtypes = {
            "ID": "int32",
            "Time": "float32",
            "Value_0": "float32",
            "Value_1": "float32",
            "Value_2": "float32",
            "Value_3": "float32",
            "Value_4": "float32",
            "Mask_0": "bool",
            "Mask_1": "bool",
            "Mask_2": "bool",
            "Mask_3": "bool",
            "Mask_4": "bool",
        }
        frame = pd.read_csv(self.csv_path, dtype=dtypes)
        self._validate_columns(frame)
        return self._to_multivariate_frame(frame)

    @staticmethod
    def _validate_columns(frame: DataFrame) -> None:
        """检查原始 CSV 是否包含任务所需列。"""
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"USHCN sparse CSV is missing columns: {missing}")

    @staticmethod
    def _to_multivariate_frame(frame: DataFrame) -> DataFrame:
        """把稀疏列格式重排为 ``(ID, Time) -> 多变量`` 的表格。"""
        data = frame.copy()
        channel_columns: dict[str, str] = {}
        for channel in range(5):
            channel_key = f"CH_{channel}"
            value_key = f"Value_{channel}"
            mask_key = f"Mask_{channel}"
            channel_columns[channel_key] = value_key
            data[channel_key] = data[value_key].where(data[mask_key])

        multivariate = data[["ID", "Time", *channel_columns]]
        multivariate = multivariate.sort_values(["ID", "Time"])
        multivariate = multivariate.set_index(["ID", "Time"])
        return multivariate.rename(columns=channel_columns)
