from __future__ import annotations

"""Loader for the preprocessed MIMIC-III event tensor."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
from pandas import DataFrame


REQUIRED_COLUMNS = {
    "UNIQUE_ID",
    "TIME_STAMP",
    "LABEL_CODE",
    "VALUENORM",
}
NUM_VARIABLES = 96


@lru_cache(maxsize=2)
def load_mimic_frame(
    csv_path: str,
    source_mtime_ns: int,
    source_size: int,
) -> DataFrame:
    """Read triplets and cache the wide irregular frame within one process."""

    del source_mtime_ns, source_size
    source = Path(csv_path)
    events = pd.read_csv(
        source,
        usecols=["UNIQUE_ID", "TIME_STAMP", "LABEL_CODE", "VALUENORM"],
        dtype={
            "UNIQUE_ID": "int32",
            "TIME_STAMP": "int16",
            "LABEL_CODE": "int16",
            "VALUENORM": "float32",
        },
    )
    missing = REQUIRED_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"MIMIC-III tensor is missing columns: {sorted(missing)}")

    events["VALUENORM"] = events["VALUENORM"].replace([float("inf"), -float("inf")], pd.NA)
    frame = events.pivot_table(
        index=["UNIQUE_ID", "TIME_STAMP"],
        columns="LABEL_CODE",
        values="VALUENORM",
        aggfunc="first",
    )
    frame = frame.reindex(columns=range(NUM_VARIABLES))
    frame.columns = [f"CH_{index}" for index in range(NUM_VARIABLES)]
    frame = frame.sort_index()
    frame.columns.name = None
    return frame.astype("float32")


class MIMICIIICompleteTensorDataset:
    """Load ``complete_tensor.csv`` into ``(UNIQUE_ID, TIME_STAMP)`` rows."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    def load(self) -> DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"MIMIC-III complete tensor not found: {self.csv_path}. "
                "Run the MIMIC-III preprocessing pipeline first."
            )
        stat = self.csv_path.stat()
        return load_mimic_frame(
            str(self.csv_path.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
        )
