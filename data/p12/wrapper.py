from __future__ import annotations

"""P12 数据集 wrapper。"""

import math

from torch.utils.data import ConcatDataset, Dataset

from ..config import DataConfig
from .task import P12Task


class P12WrapperDataset(Dataset):
    """对 P12Task 的轻量封装。"""

    def __init__(self, config: DataConfig, flag: str = "train"):
        if flag not in {"train", "val", "test", "test_all"}:
            raise ValueError(f"Unsupported split: {flag}")

        self.config = config
        self.flag = flag
        self.task = P12Task(config)

        if self._needs_length_scan():
            self._scan_irregular_lengths()

        fold = self.config.fold_index
        if flag == "test_all":
            self.dataset = ConcatDataset(list(self.task.get_all_datasets(fold)))
        else:
            self.dataset = self.task.get_dataset((fold, flag))

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def _needs_length_scan(self) -> bool:
        """判断是否需要先扫描长度。"""
        if self.config.seq_len_max_irr is None or self.config.pred_len_max_irr is None:
            return True
        if self.config.collate_fn in {"collate_fn_patch", "collate_fn_tpatch"}:
            return self.config.patch_len_max_irr is None
        return False

    def _scan_irregular_lengths(self) -> None:
        """扫描当前 fold 所有样本的最大历史/预测/patch 长度。"""
        all_dataset = ConcatDataset(list(self.task.get_all_datasets(self.config.fold_index)))
        seq_len_max_irr = 0
        pred_len_max_irr = 0
        patch_len_max_irr = 0

        seq_len = self.config.seq_len
        patch_len = self.config.patch_len

        for sample in all_dataset:
            x_mark, x, y_mark = sample.inputs
            y = sample.targets

            seq_len_max_irr = max(seq_len_max_irr, int(x.shape[0]))
            pred_len_max_irr = max(pred_len_max_irr, int(y.shape[0]))

            if self.config.collate_fn == "collate_fn_patch":
                if seq_len % patch_len != 0:
                    raise ValueError(
                        f"seq_len {seq_len} must be divisible by patch_len {patch_len}."
                    )

                n_patch = seq_len // patch_len
                n_patch_y = math.ceil(self.config.pred_len / patch_len)

                patch_i_end_previous = 0
                for i in range(n_patch):
                    observations = x_mark < ((i + 1) * patch_len / self.config.max_time_steps)
                    patch_i_end = int(observations.sum().item())
                    sample_mask = slice(patch_i_end_previous, patch_i_end)
                    patch_len_max_irr = max(patch_len_max_irr, int(len(x[sample_mask])))
                    patch_i_end_previous = patch_i_end

                patch_j_end_previous = 0
                for j in range(n_patch_y):
                    observations = y_mark < (
                        ((n_patch + j + 1) * patch_len) / self.config.max_time_steps
                    )
                    patch_j_end = int(observations.sum().item())
                    sample_mask = slice(patch_j_end_previous, patch_j_end)
                    patch_len_max_irr = max(patch_len_max_irr, int(len(y[sample_mask])))
                    patch_j_end_previous = patch_j_end

        if self.config.collate_fn == "collate_fn_patch":
            n_patch = seq_len // patch_len
            n_patch_y = math.ceil(self.config.pred_len / patch_len)
            seq_len_max_irr = max(seq_len_max_irr, patch_len_max_irr * n_patch)
            pred_len_max_irr = max(pred_len_max_irr, patch_len_max_irr * n_patch_y)

        self.config.seq_len_max_irr = seq_len_max_irr
        self.config.pred_len_max_irr = pred_len_max_irr
        if self.config.collate_fn in {"collate_fn_patch", "collate_fn_tpatch"}:
            self.config.patch_len_max_irr = patch_len_max_irr
