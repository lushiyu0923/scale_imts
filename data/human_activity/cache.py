from __future__ import annotations

"""HumanActivity 缓存文件路径工具。"""

from pathlib import Path


def get_processed_dir(dataset_root: Path) -> Path:
    """返回 processed 目录，不存在则创建。"""
    processed_dir = dataset_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def get_records_cache_path(dataset_root: Path) -> Path:
    """HumanActivity 解析后记录缓存路径。"""
    return get_processed_dir(dataset_root) / "human_activity_records.pt"
