from __future__ import annotations

"""Dataset wrapper for MIMIC-III forecasting."""

import math

from torch.utils.data import ConcatDataset, Dataset

from ..config import DataConfig
from .task import MIMICIIITask


class MIMICIIIWrapperDataset(Dataset):
    def __init__(self, config: DataConfig, flag: str = "train"):
        if flag not in {"train", "val", "test", "test_all"}:
            raise ValueError(f"Unsupported split: {flag}")

        self.config = config
        self.flag = flag
        self.task = MIMICIIITask(config)

        if self._needs_length_scan():
            self._scan_irregular_lengths()

        fold = config.fold_index
        if flag == "test_all":
            self.dataset = ConcatDataset(list(self.task.get_all_datasets(fold)))
        else:
            self.dataset = self.task.get_dataset((fold, flag))

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def _needs_length_scan(self) -> bool:
        if self.config.seq_len_max_irr is None or self.config.pred_len_max_irr is None:
            return True
        if self.config.collate_fn in {"collate_fn_patch", "collate_fn_tpatch"}:
            return self.config.patch_len_max_irr is None
        return False

    def _scan_irregular_lengths(self) -> None:
        all_dataset = ConcatDataset(list(self.task.get_all_datasets(self.config.fold_index)))
        seq_len_max_irr = 0
        pred_len_max_irr = 0
        patch_len_max_irr = 0

        for sample in all_dataset:
            x_mark, x, y_mark = sample.inputs
            y = sample.targets
            seq_len_max_irr = max(seq_len_max_irr, int(x.shape[0]))
            pred_len_max_irr = max(pred_len_max_irr, int(y.shape[0]))

            if self.config.collate_fn == "collate_fn_patch":
                if self.config.seq_len % self.config.patch_len != 0:
                    raise ValueError(
                        f"seq_len {self.config.seq_len} must be divisible by "
                        f"patch_len {self.config.patch_len}."
                    )
                n_patch = self.config.seq_len // self.config.patch_len
                n_patch_y = math.ceil(self.config.pred_len / self.config.patch_len)

                previous = 0
                for i in range(n_patch):
                    end = int((x_mark < ((i + 1) * self.config.patch_len / self.config.max_time_steps)).sum())
                    patch_len_max_irr = max(patch_len_max_irr, end - previous)
                    previous = end

                previous = 0
                for j in range(n_patch_y):
                    end = int((y_mark < (((n_patch + j + 1) * self.config.patch_len) / self.config.max_time_steps)).sum())
                    patch_len_max_irr = max(patch_len_max_irr, end - previous)
                    previous = end

        if self.config.collate_fn == "collate_fn_patch":
            n_patch = self.config.seq_len // self.config.patch_len
            n_patch_y = math.ceil(self.config.pred_len / self.config.patch_len)
            seq_len_max_irr = max(seq_len_max_irr, patch_len_max_irr * n_patch)
            pred_len_max_irr = max(pred_len_max_irr, patch_len_max_irr * n_patch_y)

        self.config.seq_len_max_irr = seq_len_max_irr
        self.config.pred_len_max_irr = pred_len_max_irr
        if self.config.collate_fn in {"collate_fn_patch", "collate_fn_tpatch"}:
            self.config.patch_len_max_irr = patch_len_max_irr
