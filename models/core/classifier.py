from __future__ import annotations

"""ScaleIMTS 分类包装器。

不修改主干 ``Model``，只复用它对历史序列的编码和融合路径，
再接一个线性分类头。分类没有未来预测点，因此用每个样本
最后一个有效观测时间构造长度为 1 的 dummy query。
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ScaleRoutingContext
from .model import Model


class ScaleIMTSClassifier(nn.Module):
    """在冻结结构的 ScaleIMTS 编码器上做样本级分类。"""

    def __init__(self, backbone: Model, n_classes: int, dropout: float = 0.1):
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")
        self.backbone = backbone
        self.n_classes = n_classes
        fusion_type = getattr(backbone.fusion_router, "fusion_type", "concat")
        feature_dim = backbone.fusion_router.fusion_hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_classes),
        )

    def _dummy_y_mark(self, x_mark: Tensor, x_mask: Tensor) -> Tensor:
        """用最后一个有效时间戳构造分类 query。"""

        if x_mark.dim() == 2:
            time = x_mark.unsqueeze(-1)
        else:
            time = x_mark
        valid = x_mask.any(dim=-1)
        filled = torch.where(valid, time.squeeze(-1), torch.zeros_like(time.squeeze(-1)))
        last_time = filled.max(dim=1, keepdim=True).values.unsqueeze(-1)
        return last_time

    def encode(self, x: Tensor, x_mark: Tensor, x_mask: Tensor, y_mark: Tensor | None = None) -> Tensor:
        """复用 ScaleIMTS 的 align / branch / fusion，返回 fused 特征。"""

        backbone = self.backbone
        if y_mark is None:
            y_mark = self._dummy_y_mark(x_mark, x_mask)
        aligned_batch = backbone._align_structure(x=x, x_mark=x_mark, x_mask=x_mask)
        future_queries = backbone._encode_future_queries(y_mark=y_mark)
        irregular_features = backbone._run_optional_irregular_branch(aligned_batch.irregular)
        if backbone.foundation_branch is None:
            reduced_foundation_features = backbone._null_branch_features(
                x,
                x_mask,
                backbone.config.foundation_reduced_dim,
            )
        elif backbone.foundation_branch.scale_mode == "none":
            foundation_features = backbone._run_foundation_branch(aligned_batch.foundation)
            reduced_foundation_features = backbone._reduce_foundation_features(
                foundation_features=foundation_features,
                irregular_features=irregular_features,
                future_queries=future_queries,
            )
        else:
            scale_context = ScaleRoutingContext(
                future_queries=future_queries,
                irregular_global_features=(
                    irregular_features.global_features if backbone.use_irregular_branch else None
                ),
            )
            foundation_features = backbone._run_foundation_branch(
                aligned_batch.foundation,
                scale_context=scale_context,
            )
            reduced_foundation_features = backbone._reduce_foundation_features(
                foundation_features=foundation_features,
                irregular_features=irregular_features,
                future_queries=future_queries,
            )
        return backbone._fuse_branch_features(
            foundation_features=reduced_foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )

    def _pool_fused(self, fused_features: Tensor) -> Tensor:
        if fused_features.ndim == 4:
            pooled = fused_features.mean(dim=2)
            return pooled.mean(dim=1)
        if fused_features.ndim == 3:
            return fused_features.mean(dim=1)
        return fused_features

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor,
        x_mask: Tensor,
        **kwargs,
    ) -> dict[str, Tensor | None]:
        labels: Tensor | None = kwargs.get("y")
        y_mark: Tensor | None = kwargs.get("y_mark")
        fused_features = self.encode(x=x, x_mark=x_mark, x_mask=x_mask, y_mark=y_mark)
        logits = self.classifier(self._pool_fused(fused_features))
        if labels is not None and labels.dtype != torch.long:
            labels = labels.long()
        if labels is not None and labels.ndim > 1:
            labels = labels.view(labels.shape[0], -1)[:, 0]
        return {
            "pred": logits,
            "true": labels,
            "mask": None,
        }
