from __future__ import annotations

"""通用时间编码模块。"""

import torch
import torch.nn as nn
from torch import Tensor


class TimeFeatureEncoder(nn.Module):
    """APN 风格的可学习时间编码器。

    输出由两部分组成：
    - 一维线性时间项；
    - 多维周期时间项。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2 for APN-style time encoding.")
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, hidden_dim - 1)

    def forward(self, time_values: Tensor) -> Tensor:
        """把形如 ``[..., 1]`` 的时间戳映射为 ``[..., hidden_dim]``。"""
        linear_part = self.linear(time_values)
        periodic_part = torch.sin(self.periodic(time_values))
        return torch.cat([linear_part, periodic_part], dim=-1)
