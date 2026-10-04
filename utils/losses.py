from __future__ import annotations

"""损失函数集合。"""

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """带 mask 的 MSE。

    只有 ``mask == 1`` 的位置才会参与损失，
    适合不规则时间序列里存在 padding 或缺失的场景。
    """

    def forward(self, pred, true, mask=None, **kwargs):
        if true is None:
            raise ValueError("MaskedMSELoss requires `true` in the model output.")
        if mask is None:
            mask = torch.ones_like(true, device=true.device)

        residual = (pred - true) * mask
        num_eval = mask.sum()
        # 分母至少为 1，避免全空 mask 时出现除零错误。
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return {"loss": loss}


class ClassificationCELoss(nn.Module):
    """样本级交叉熵，供 IMTS 分类任务使用。"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, true, mask=None, **kwargs):
        if true is None:
            raise ValueError("ClassificationCELoss requires `true` in the model output.")
        labels = true.long()
        if labels.ndim > 1:
            labels = labels.view(labels.shape[0], -1)[:, 0]
        return {"loss": self.criterion(pred, labels)}
