from __future__ import annotations

"""foundation 路的输入对齐与 token 化模块。"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import BranchAlignedBatch
from .time_encoders import TimeFeatureEncoder


class FoundationValueEmbedding(nn.Module):
    """把 ``[数值, 是否观测]`` 二元组投影到 PLM 隐空间。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(2, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class FoundationVariableEmbedding(nn.Module):
    """变量提示向量（variable prompt）的嵌入层。"""

    def __init__(self, n_var: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(n_var, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x.long())


class FoundationIndVarPromptEmbedding(nn.Module):
    """迁移自 ISTS-PLM 的输入嵌入逻辑。

    这个模块负责把：
    - 数值；
    - 观测 mask；
    - 连续时间；
    - 变量 prompt
    组合成 foundation backbone 可直接消费的 token 序列。
    """

    def __init__(self, n_var: int, d_model: int, dropout: float = 0.1, use_te: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_var = n_var
        self.use_te = use_te
        self.time_embedding = TimeFeatureEncoder(hidden_dim=d_model)
        self.value_embedding = FoundationValueEmbedding(d_model=d_model)
        self.variable_embedding = FoundationVariableEmbedding(n_var=n_var, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt: Tensor, x: Tensor, x_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """输出 token、变量 prompt，以及与 token 对齐的 mask。"""
        batch_size, seq_len, n_var = x.shape
        time_emb = self.time_embedding(tt.unsqueeze(dim=-1)) if self.use_te else None
        x_mask_expanded = x_mask.unsqueeze(dim=-1)

        variable_ids = torch.arange(n_var, device=x.device).view(1, 1, -1).repeat(batch_size, 1, 1)
        variable_prompts = self.variable_embedding(variable_ids)

        x_value = x.unsqueeze(dim=-1)
        x_int = torch.cat([x_value, x_mask_expanded], dim=-1)
        value_emb = self.value_embedding(x_int)

        if self.use_te:
            # 只在真实观测位置叠加时间特征，避免 padding 时间污染 token。
            token_features = x_mask_expanded * time_emb + value_emb
        else:
            token_features = value_emb

        token_features = torch.cat([variable_prompts, token_features], dim=1)
        token_features = token_features.permute(0, 2, 1, 3).reshape(
            batch_size * n_var, seq_len + 1, self.d_model
        )

        token_mask = x_mask.permute(0, 2, 1).reshape(batch_size * n_var, seq_len, 1)
        token_mask = torch.cat([torch.ones_like(token_mask[:, :1]), token_mask], dim=1)

        return self.dropout(token_features), variable_prompts, token_mask


class FoundationAligner(nn.Module):
    """foundation 路对齐器。

    负责把原始不规则多变量序列整理成 foundation backbone 需要的 token 表示。
    """

    def __init__(
        self,
        n_var: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_time_features: bool = True,
    ):
        super().__init__()
        self.embedding = FoundationIndVarPromptEmbedding(
            n_var=n_var,
            d_model=hidden_dim,
            dropout=dropout,
            use_te=use_time_features,
        )

    def forward(self, x: Tensor, x_mark: Tensor, x_mask: Tensor) -> BranchAlignedBatch:
        """返回 foundation 路专用的 ``BranchAlignedBatch``。"""
        time_per_variable = x_mark.repeat(1, 1, x.shape[-1])
        tokens, variable_prompts, token_mask = self.embedding(
            tt=time_per_variable,
            x=x,
            x_mask=x_mask,
        )
        return BranchAlignedBatch(
            values=x,
            time=x_mark,
            mask=x_mask,
            tokens=tokens,
            token_mask=token_mask,
            variable_prompts=variable_prompts,
        )
