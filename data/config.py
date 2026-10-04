from __future__ import annotations

"""数据流水线配置定义。"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """不规则多变量时间序列数据配置。

    这个配置会被具体数据集 wrapper 读取，并在扫描样本长度后回填
    ``seq_len_max_irr / pred_len_max_irr / patch_len_max_irr`` 等字段。
    """

    dataset_name: str
    dataset_root_path: str
    raw_data_files: list[str] = field(default_factory=list)
    raw_csv_name: str = "small_chunked_sporadic.csv"
    seq_len: int = 150
    pred_len: int = 3
    label_len: int = 0
    batch_size: int = 8
    num_workers: int = 0
    collate_fn: str = "collate_fn"
    missing_rate: float = 0.0
    perturbation_mode: str = "none"
    perturbation_rate: float = 0.0
    perturbation_seed: int = 2026
    perturbation_split: str = "test"
    perturbation_min_observations: int = 1
    perturbation_local_scale_contrast: float = 0.75
    patch_len: int = 12
    train_val_loader_shuffle: bool | None = None
    train_val_loader_drop_last: bool | None = None
    normalize_time: bool = True
    random_seed: int = 432
    fold_index: int = 0
    num_folds: int = 5
    test_size: float = 0.1
    valid_size: float = 0.1
    max_time_steps: int = 200
    enc_in: int = 5
    c_out: int = 5
    features: str = "M"

    # Filled by the wrapper after scanning train/val/test samples.
    seq_len_max_irr: int | None = None
    pred_len_max_irr: int | None = None
    patch_len_max_irr: int | None = None

    @property
    def dataset_root(self) -> Path:
        """数据集根目录。"""
        return Path(self.dataset_root_path)

    @property
    def raw_csv_path(self) -> Path:
        """单个 CSV 数据集的原始文件路径。"""
        return self.dataset_root / self.raw_csv_name

    @property
    def raw_file_paths(self) -> list[Path]:
        """多文件数据集的原始文件路径列表。"""
        if self.raw_data_files:
            return [self.dataset_root / name for name in self.raw_data_files]
        return [self.raw_csv_path]
