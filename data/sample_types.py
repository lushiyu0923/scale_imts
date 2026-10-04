from __future__ import annotations

"""不规则预测任务里单样本的数据结构定义。"""

from typing import NamedTuple

from torch import Tensor


class Inputs(NamedTuple):
    """单个预测样本的输入部分。"""

    t: Tensor
    x: Tensor
    t_target: Tensor


class Sample(NamedTuple):
    """单个不规则预测样本。"""

    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]
