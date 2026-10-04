from __future__ import annotations

"""P12 原始 tar.gz 读取与缓存。"""

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from .cache import get_timeseries_cache_path


GENERAL_DESCRIPTORS = {
    "Age": int,
    "Gender": int,
    "Height": float,
    "ICUType": int,
    "Weight": float,
}

VARIABLES = {
    "Albumin": np.nan,
    "ALP": np.nan,
    "ALT": np.nan,
    "AST": np.nan,
    "Bilirubin": np.nan,
    "BUN": np.nan,
    "Cholesterol": np.nan,
    "Creatinine": np.nan,
    "DiasABP": np.nan,
    "FiO2": np.nan,
    "GCS": np.nan,
    "Glucose": np.nan,
    "HCO3": np.nan,
    "HCT": np.nan,
    "HR": np.nan,
    "K": np.nan,
    "Lactate": np.nan,
    "Mg": np.nan,
    "MAP": np.nan,
    "MechVent": np.nan,
    "Na": np.nan,
    "NIDiasABP": np.nan,
    "NIMAP": np.nan,
    "NISysABP": np.nan,
    "PaCO2": np.nan,
    "PaO2": np.nan,
    "pH": np.nan,
    "Platelets": np.nan,
    "RespRate": np.nan,
    "SaO2": np.nan,
    "SysABP": np.nan,
    "Temp": np.nan,
    "TroponinI": np.nan,
    "TroponinT": np.nan,
    "Urine": np.nan,
    "WBC": np.nan,
}

COUNTS = {key: 0 for key in VARIABLES}


def read_physionet_record(record_bytes: bytes) -> tuple[int, dict[str, float], DataFrame]:
    """解析单个 PhysioNet 记录文件。"""
    lines = iter(record_bytes.splitlines())
    _ = next(lines)

    general_descriptors: dict[str, float] = {}
    record_id = None

    timeseries = VARIABLES.copy()
    counts = COUNTS.copy()
    previous_hour = -1
    first = True
    row_hours: list[int] = []
    row_values: list[list[float]] = []

    def flush_current_hour(hour: int, values: dict[str, float], current_counts: dict[str, int]) -> None:
        """把同一小时内的多次观测聚合为一行。"""
        row_hours.append(hour)
        row_values.append(
            [
                float(values[key] / current_counts[key]) if current_counts[key] > 0 else np.nan
                for key in VARIABLES
            ]
        )

    for raw_line in lines:
        time_token, parameter_token, value_token = (token.strip() for token in raw_line.split(b","))
        hours, minutes = time_token.split(b":")
        time_minutes = int(hours) * 60 + int(minutes)
        hour_bucket = time_minutes // 60
        parameter = parameter_token.decode("utf-8")

        if parameter == "RecordID":
            record_id = int(value_token.decode())
            continue

        if parameter in GENERAL_DESCRIPTORS and time_minutes == 0:
            value = GENERAL_DESCRIPTORS[parameter](value_token.decode())
            general_descriptors[parameter] = value if value > -1 else np.nan
            continue

        value = float(value_token.decode())
        if first or previous_hour == hour_bucket:
            if first:
                previous_hour = hour_bucket
                first = False
            if parameter not in timeseries:
                continue
            if np.isnan(timeseries[parameter]):
                timeseries[parameter] = 0.0
                timeseries[parameter] += value
                counts[parameter] += 1
        else:
            flush_current_hour(previous_hour, timeseries, counts)
            previous_hour = hour_bucket
            timeseries = VARIABLES.copy()
            counts = COUNTS.copy()
            if parameter in timeseries:
                timeseries[parameter] = value
                counts[parameter] = 1

    if not first:
        flush_current_hour(previous_hour, timeseries, counts)

    if record_id is None:
        raise ValueError("RecordID not found in PhysioNet record.")
    for key in GENERAL_DESCRIPTORS:
        if key not in general_descriptors:
            raise ValueError(f"Incomplete PhysioNet metadata: missing {key}")

    frame = pd.DataFrame(row_values, index=np.array(row_hours, dtype=np.int32), columns=list(VARIABLES))
    frame.index.name = "Time"
    return record_id, general_descriptors, frame


class Physionet2012RawDataset:
    """读取并缓存 P12 原始 tar.gz 到多变量稀疏表格。"""

    def __init__(self, dataset_root: Path, raw_file_paths: list[Path]):
        self.dataset_root = dataset_root
        self.raw_file_paths = raw_file_paths

    def load(self) -> DataFrame:
        cache_path = get_timeseries_cache_path(self.dataset_root)
        if cache_path.exists():
            return pd.read_pickle(cache_path)

        timeseries = self._read_from_tarballs()
        timeseries.to_pickle(cache_path)
        return timeseries

    def _read_from_tarballs(self) -> DataFrame:
        """遍历所有 tar.gz，收集所有病人的观测记录。"""
        all_record_ids: list[int] = []
        all_timestamps: list[int] = []
        all_values: list[np.ndarray] = []

        for raw_path in self.raw_file_paths:
            with tarfile.open(raw_path, "r:gz") as archive:
                for member in archive.getmembers():
                    if not member.isreg():
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    record_bytes = extracted.read()
                    record_id, _, observations = read_physionet_record(record_bytes)
                    if len(observations) == 0:
                        continue
                    all_record_ids.extend([record_id] * len(observations))
                    all_timestamps.extend(observations.index.to_list())
                    all_values.append(observations.to_numpy(dtype=np.float32, copy=False))

        if not all_values:
            raise ValueError("No valid PhysioNet observations were loaded from the P12 tarballs.")

        values = np.vstack(all_values)
        time_series_df = pd.DataFrame(values, columns=list(VARIABLES))
        time_series_df.set_index(
            pd.MultiIndex.from_arrays(
                (np.array(all_record_ids, dtype=np.int64), np.array(all_timestamps, dtype=np.int32)),
                names=("RecordID", "Time"),
            ),
            inplace=True,
        )
        time_series_df.columns.name = None
        return time_series_df
