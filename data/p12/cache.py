from __future__ import annotations

"""P12 缓存文件路径工具。"""

from pathlib import Path


def get_processed_dir(dataset_root: Path) -> Path:
    """返回 processed 目录，不存在则创建。"""
    processed_dir = dataset_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def get_timeseries_cache_path(dataset_root: Path) -> Path:
    """P12 预处理后时间序列表缓存路径。"""
    return get_processed_dir(dataset_root) / "p12_timeseries.pkl"
