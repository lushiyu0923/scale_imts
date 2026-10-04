from __future__ import annotations

"""P12 的 batch 组装逻辑。"""

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..config import DataConfig
from ..sample_types import Sample


def fix_nan_x_mark(x_mark: Tensor, seq_len: int, total_time_steps: int) -> Tensor:
    """为 padding 位置构造稳定的占位时间。"""
    batch_size, seq_len_max_irr, _ = x_mark.shape
    indices = torch.linspace(
        start=seq_len / total_time_steps - 2 * 0.01,
        end=seq_len / total_time_steps - 0.001,
        steps=seq_len_max_irr,
        device=x_mark.device,
    ).view(1, -1, 1).repeat(batch_size, 1, 1)
    nan_mask = torch.isnan(x_mark)
    x_mark[nan_mask] = indices[nan_mask]
    return x_mark


def fix_nan_y_mark(y_mark: Tensor) -> Tensor:
    """为预测窗口 padding 位置构造稳定的占位时间。"""
    batch_size, pred_len, _ = y_mark.shape
    indices = torch.linspace(
        start=1 - 2 * 0.01,
        end=1 - 0.001,
        steps=pred_len,
        device=y_mark.device,
    ).view(1, -1, 1).repeat(batch_size, 1, 1)
    nan_mask = torch.isnan(y_mark)
    y_mark[nan_mask] = indices[nan_mask]
    return y_mark


def _apply_missing_rate(xs: Tensor, x_masks: Tensor, missing_rate: float) -> None:
    """按比例随机额外抹掉部分可观测输入。"""
    if missing_rate <= 0:
        return
    flat_mask = x_masks.view(-1)
    flat_x = xs.view(-1)
    available_flat_indices = torch.where(flat_mask == 1)[0]
    num_available = available_flat_indices.size(0)
    num_to_mask = int(missing_rate * num_available)
    if num_to_mask <= 0:
        return
    perm = torch.randperm(num_available, device=available_flat_indices.device)
    selected_flat = available_flat_indices[perm[:num_to_mask]]
    flat_x[selected_flat] = torch.nan
    flat_mask[selected_flat] = 0


def build_collate_fn(config: DataConfig) -> Callable[[list[Sample]], dict[str, Tensor]]:
    def _collate(batch: list[Sample]) -> dict[str, Tensor]:
        """标准 P12 collate。"""
        if config.seq_len_max_irr is None or config.pred_len_max_irr is None:
            raise ValueError("Irregular sequence lengths were not initialized before collate.")

        xs: list[Tensor] = []
        ys: list[Tensor] = []
        x_marks: list[Tensor] = []
        y_marks: list[Tensor] = []
        x_masks: list[Tensor] = []
        y_masks: list[Tensor] = []
        sample_ids: list[int] = []

        for sample in batch:
            x_mark, x, y_mark = sample.inputs
            y = sample.targets
            sample_id = sample.key

            x_mask = x.isfinite()
            y_mask = y.isfinite()

            xs.append(x)
            x_marks.append(x_mark)
            x_masks.append(x_mask)
            ys.append(y)
            y_marks.append(y_mark)
            y_masks.append(y_mask)
            sample_ids.append(sample_id)

        enc_in = xs[0].shape[-1]

        xs.append(torch.zeros(config.seq_len_max_irr, enc_in))
        x_marks.append(torch.zeros(config.seq_len_max_irr))
        x_masks.append(torch.zeros(config.seq_len_max_irr, enc_in))
        ys.append(torch.zeros(config.pred_len_max_irr, enc_in))
        y_marks.append(torch.zeros(config.pred_len_max_irr))
        y_masks.append(torch.zeros(config.pred_len_max_irr, enc_in))

        xs_padded = pad_sequence(xs, batch_first=True, padding_value=float("nan"))[:-1]
        x_marks_padded = pad_sequence(x_marks, batch_first=True, padding_value=float("nan"))[:-1]
        x_masks_padded = pad_sequence(x_masks, batch_first=True)[:-1]
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=float("nan"))[:-1]
        y_marks_padded = pad_sequence(y_marks, batch_first=True, padding_value=float("nan"))[:-1]
        y_masks_padded = pad_sequence(y_masks, batch_first=True)[:-1]

        _apply_missing_rate(xs_padded, x_masks_padded, config.missing_rate)

        return {
            "x": torch.nan_to_num(xs_padded),
            "x_mark": fix_nan_x_mark(
                x_marks_padded.unsqueeze(-1),
                seq_len=config.seq_len,
                total_time_steps=config.max_time_steps,
            ).float(),
            "x_mask": x_masks_padded.float(),
            "y": torch.nan_to_num(ys_padded),
            "y_mark": fix_nan_y_mark(y_marks_padded.unsqueeze(-1)).float(),
            "y_mask": y_masks_padded.float(),
            "sample_ID": torch.tensor(sample_ids).float(),
        }

    return _collate


def build_collate_fn_patch(config: DataConfig) -> Callable[[list[Sample]], dict[str, Tensor]]:
    def _collate(batch: list[Sample]) -> dict[str, Tensor]:
        """patch 版本 P12 collate。"""
        if config.patch_len_max_irr is None:
            raise ValueError("patch_len_max_irr was not initialized before patch collate.")

        patch_len_max_irr = config.patch_len_max_irr
        patch_len = config.patch_len
        seq_len = config.seq_len
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len {seq_len} must be divisible by patch_len {patch_len}.")

        n_patch = seq_len // patch_len
        n_patch_y = math.ceil(config.pred_len / patch_len)

        xs: list[Tensor] = []
        ys: list[Tensor] = []
        x_marks: list[Tensor] = []
        y_marks: list[Tensor] = []
        x_masks: list[Tensor] = []
        y_masks: list[Tensor] = []
        sample_ids: list[int] = []

        for sample in batch:
            x_mark, x, y_mark = sample.inputs
            y = sample.targets

            patch_i_end_previous = 0
            for i in range(n_patch):
                observations = x_mark < ((i + 1) * patch_len / config.max_time_steps)
                patch_i_end = int(observations.sum().item())
                sample_mask = slice(patch_i_end_previous, patch_i_end)
                x_patch = x[sample_mask]
                if len(x_patch) == 0:
                    xs.append(torch.full((1, x.shape[-1]), fill_value=float("nan"), device=x.device))
                    x_marks.append(torch.zeros((1), device=x.device))
                    x_masks.append(torch.zeros((1, x.shape[-1]), device=x.device))
                else:
                    xs.append(x_patch)
                    x_marks.append(x_mark[sample_mask])
                    x_masks.append(x_patch.isfinite().int())
                patch_i_end_previous = patch_i_end

            patch_j_end_previous = 0
            for j in range(n_patch_y):
                observations = y_mark < (((n_patch + j + 1) * patch_len) / config.max_time_steps)
                patch_j_end = int(observations.sum().item())
                sample_mask = slice(patch_j_end_previous, patch_j_end)
                y_patch = y[sample_mask]
                if len(y_patch) == 0:
                    ys.append(torch.full((1, y.shape[-1]), fill_value=float("nan"), device=y.device))
                    y_marks.append(torch.zeros((1), device=y.device))
                    y_masks.append(torch.zeros((1, y.shape[-1]), device=y.device))
                else:
                    ys.append(y_patch)
                    y_marks.append(y_mark[sample_mask])
                    y_masks.append(y_patch.isfinite().int())
                patch_j_end_previous = patch_j_end

            sample_ids.append(sample.key)

        enc_in = xs[0].shape[-1]
        xs.append(torch.zeros(patch_len_max_irr, enc_in))
        x_marks.append(torch.zeros(patch_len_max_irr))
        x_masks.append(torch.zeros(patch_len_max_irr, enc_in))
        ys.append(torch.zeros(patch_len_max_irr, enc_in))
        y_marks.append(torch.zeros(patch_len_max_irr))
        y_masks.append(torch.zeros(patch_len_max_irr, enc_in))

        xs_padded = pad_sequence(xs, batch_first=True, padding_value=float("nan"))[:-1]
        x_marks_padded = pad_sequence(x_marks, batch_first=True)[:-1]
        x_masks_padded = pad_sequence(x_masks, batch_first=True)[:-1]
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=float("nan"))[:-1]
        y_marks_padded = pad_sequence(y_marks, batch_first=True)[:-1]
        y_masks_padded = pad_sequence(y_masks, batch_first=True)[:-1]

        _apply_missing_rate(xs_padded, x_masks_padded, config.missing_rate)

        return {
            "x": torch.nan_to_num(xs_padded.view(-1, patch_len_max_irr * n_patch, enc_in)),
            "x_mark": x_marks_padded.view(-1, patch_len_max_irr * n_patch).unsqueeze(-1).float(),
            "x_mask": x_masks_padded.view(-1, patch_len_max_irr * n_patch, enc_in).float(),
            "y": torch.nan_to_num(ys_padded.view(-1, patch_len_max_irr * n_patch_y, enc_in)),
            "y_mark": y_marks_padded.view(-1, patch_len_max_irr * n_patch_y).unsqueeze(-1).float(),
            "y_mask": y_masks_padded.view(-1, patch_len_max_irr * n_patch_y, enc_in).float(),
            "sample_ID": torch.tensor(sample_ids).float(),
        }

    return _collate


def build_collate_fn_tpatch(config: DataConfig) -> Callable[[list[Sample]], dict[str, Tensor]]:
    def _collate(batch: list[Sample]) -> dict[str, Tensor]:
        """tPatch 风格 P12 collate。"""
        patch_len = config.patch_len
        seq_len = config.seq_len
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len {seq_len} must be divisible by patch_len {patch_len}.")

        n_patch = seq_len // patch_len
        n_patch_y = math.ceil(config.pred_len / patch_len)

        xs: list[Tensor] = []
        ys: list[Tensor] = []
        x_marks: list[Tensor] = []
        y_marks: list[Tensor] = []
        x_masks: list[Tensor] = []
        y_masks: list[Tensor] = []
        sample_ids: list[int] = []

        for sample in batch:
            x_mark, x, y_mark = sample.inputs
            y = sample.targets

            patch_i_end_previous = 0
            for i in range(n_patch):
                observations = x_mark < ((i + 1) * patch_len / config.max_time_steps)
                patch_i_end = int(observations.sum().item())
                sample_mask = slice(patch_i_end_previous, patch_i_end)
                x_patch = x[sample_mask]
                for variable in range(x_patch.shape[-1]):
                    x_patch_variable = x_patch[:, variable]
                    non_zero_mask = x_patch_variable.isfinite()
                    x_patch_non_zero = x_patch_variable[non_zero_mask]
                    if len(x_patch_variable) == 0:
                        xs.append(torch.full((1,), fill_value=float("nan"), device=x.device))
                        x_marks.append(torch.zeros((1), device=x.device))
                        x_masks.append(torch.zeros((1), device=x.device))
                    else:
                        xs.append(x_patch_non_zero)
                        x_marks.append(x_mark[sample_mask][non_zero_mask])
                        x_masks.append(x_patch_non_zero.isfinite().int())
                patch_i_end_previous = patch_i_end

            patch_j_end_previous = 0
            for j in range(n_patch_y):
                observations = y_mark < (((n_patch + j + 1) * patch_len) / config.max_time_steps)
                patch_j_end = int(observations.sum().item())
                sample_mask = slice(patch_j_end_previous, patch_j_end)
                y_patch = y[sample_mask]
                for variable in range(y_patch.shape[-1]):
                    y_patch_variable = y_patch[:, variable]
                    non_zero_mask = y_patch_variable.isfinite()
                    y_patch_non_zero = y_patch_variable[non_zero_mask]
                    if len(y_patch_variable) == 0:
                        ys.append(torch.full((1,), fill_value=float("nan"), device=y.device))
                        y_marks.append(torch.zeros((1), device=y.device))
                        y_masks.append(torch.zeros((1), device=y.device))
                    else:
                        ys.append(y_patch_non_zero)
                        y_marks.append(y_mark[sample_mask][non_zero_mask])
                        y_masks.append(y_patch_non_zero.isfinite().int())
                patch_j_end_previous = patch_j_end

            sample_ids.append(sample.key)

        xs_padded = pad_sequence(xs, batch_first=True, padding_value=float("nan"))
        x_marks_padded = pad_sequence(x_marks, batch_first=True)
        x_masks_padded = pad_sequence(x_masks, batch_first=True)
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=float("nan"))
        y_marks_padded = pad_sequence(y_marks, batch_first=True)
        y_masks_padded = pad_sequence(y_masks, batch_first=True)

        _apply_missing_rate(xs_padded, x_masks_padded, config.missing_rate)

        return {
            "x": torch.nan_to_num(xs_padded),
            "x_mark": x_marks_padded.unsqueeze(-1).float(),
            "x_mask": x_masks_padded.float(),
            "y": torch.nan_to_num(ys_padded),
            "y_mark": y_marks_padded.unsqueeze(-1).float(),
            "y_mask": y_masks_padded.float(),
            "sample_ID": torch.tensor(sample_ids).float(),
        }

    return _collate


COLLATE_BUILDERS = {
    "collate_fn": build_collate_fn,
    "collate_fn_patch": build_collate_fn_patch,
    "collate_fn_tpatch": build_collate_fn_tpatch,
}
