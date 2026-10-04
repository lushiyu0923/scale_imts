from __future__ import annotations

"""ScaleIMTS 最外层装配器。

这里把：
1. foundation aligner / branch；
2. irregular aligner / branch；
3. future time encoder；
4. fusion 与 prediction
组织成一条完整前向路径。
"""

import torch
import torch.nn as nn
from torch import Tensor

from .branches import FoundationBranch, FoundationReducer, FusionRouter, IrregularBranch, PredictionHead
from .config import BranchAlignedBatch, BranchFeatures, DualBranchAlignedBatch, ScaleIMTSConfig, ScaleRoutingContext
from .foundation_aligners import FoundationAligner
from .irregular_aligners import IrregularAligner
from .time_encoders import TimeFeatureEncoder


class Model(nn.Module):
    """ScaleIMTS 双路径模型：foundation 路 + irregular 路。"""

    def __init__(self, config: ScaleIMTSConfig):
        super().__init__()
        self.config = config
        self.enc_in = config.enc_in
        self.c_out = config.c_out
        self.seq_len_max_irr = config.seq_len_max_irr
        self.pred_len_max_irr = config.pred_len_max_irr
        self.features = config.features
        self.use_time_features = config.use_time_features
        self.ablation_variant = (config.ablation_variant or "full").lower()
        self.use_foundation_branch = self.ablation_variant != "irregular_only"
        self.use_irregular_branch = self.ablation_variant != "wo_irregular_dynamics"

        self.time_encoder = TimeFeatureEncoder(hidden_dim=config.hidden_dim)
        self.foundation_aligner = FoundationAligner(
            n_var=config.enc_in,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            use_time_features=config.use_time_features,
        )
        self.irregular_aligner = IrregularAligner(
            patch_len=config.patch_len,
            max_time_steps=config.max_time_steps,
        )
        self.foundation_branch = FoundationBranch.from_config(config) if self.use_foundation_branch else None
        self.foundation_reducer = FoundationReducer(
            foundation_hidden_dim=config.hidden_dim,
            irregular_hidden_dim=config.irregular_hidden_dim,
            reduced_dim=config.foundation_reduced_dim,
            gate_hidden_dim=config.foundation_gate_hidden_dim,
            dropout=config.dropout,
        )
        self.irregular_branch = (
            IrregularBranch(
                enc_in=config.enc_in,
                hidden_dim=config.hidden_dim,
                irregular_hidden_dim=config.irregular_hidden_dim,
                irregular_time_dim=config.irregular_time_dim,
                irregular_node_dim=config.irregular_node_dim,
                patch_len=config.patch_len,
                max_time_steps=config.max_time_steps,
                num_layers=config.num_hidden_layers,
                dropout=config.dropout,
            )
            if self.use_irregular_branch
            else None
        )
        fusion_type = config.fusion_type
        if self.ablation_variant == "concat_fusion":
            fusion_type = "concat"
        elif self.ablation_variant == "wo_foundation_refinement":
            fusion_type = "add_i2f_var"
        self.fusion_router = FusionRouter(
            hidden_dim=config.hidden_dim,
            foundation_reduced_dim=config.foundation_reduced_dim,
            irregular_hidden_dim=config.irregular_hidden_dim,
            fusion_type=fusion_type,
            fusion_hidden_dim=config.fusion_hidden_dim,
            film_hidden_dim=config.film_hidden_dim,
            dropout=config.dropout,
        )
        prediction_hidden_dim = config.fusion_hidden_dim
        self.prediction_head = PredictionHead(
            hidden_dim=prediction_hidden_dim,
            c_out=config.c_out,
            dropout=config.dropout,
            num_variables=config.enc_in,
        )

    def _validate_inputs(
        self,
        x: Tensor,
        x_mark: Tensor,
        x_mask: Tensor,
        y_mark: Tensor,
    ) -> None:
        """检查输入张量的长度和通道数是否与配置一致。"""
        if x.shape[1] != self.seq_len_max_irr:
            raise ValueError(
                f"Expected x.shape[1] == {self.seq_len_max_irr}, got {x.shape[1]}."
            )
        if x.shape[-1] != self.enc_in:
            raise ValueError(
                f"Expected x.shape[-1] == {self.enc_in}, got {x.shape[-1]}."
            )
        if x_mark.shape[1] != self.seq_len_max_irr:
            raise ValueError(
                f"Expected x_mark.shape[1] == {self.seq_len_max_irr}, got {x_mark.shape[1]}."
            )
        if x_mask.shape != x.shape:
            raise ValueError(
                f"Expected x_mask.shape == x.shape == {tuple(x.shape)}, got {tuple(x_mask.shape)}."
            )
        if y_mark.shape[1] != self.pred_len_max_irr:
            raise ValueError(
                f"Expected y_mark.shape[1] == {self.pred_len_max_irr}, got {y_mark.shape[1]}."
            )

    def _align_structure(self, x: Tensor, x_mark: Tensor, x_mask: Tensor) -> DualBranchAlignedBatch:
        """分别调用两条分支的 aligner。"""
        foundation_batch = self.foundation_aligner(x=x, x_mark=x_mark, x_mask=x_mask)
        irregular_batch = self.irregular_aligner(x=x, x_mark=x_mark, x_mask=x_mask)
        return DualBranchAlignedBatch(
            foundation=foundation_batch,
            irregular=irregular_batch,
        )

    def _run_foundation_branch(
        self,
        aligned_batch: BranchAlignedBatch,
        scale_context: ScaleRoutingContext | None = None,
    ) -> BranchFeatures:
        if self.foundation_branch is None:
            raise RuntimeError("Foundation branch is disabled for this ablation variant.")
        return self.foundation_branch(aligned_batch, scale_context=scale_context)

    def _run_irregular_branch(self, aligned_batch: BranchAlignedBatch) -> BranchFeatures:
        if self.irregular_branch is None:
            return self._null_branch_features(
                aligned_batch.values,
                aligned_batch.mask,
                self.config.irregular_hidden_dim,
            )
        return self.irregular_branch(aligned_batch)

    @staticmethod
    def _null_branch_features(x: Tensor, x_mask: Tensor, hidden_dim: int) -> BranchFeatures:
        batch_size, _, n_var = x.shape
        variable_mask = x_mask.sum(dim=1).gt(0)
        return BranchFeatures(
            sequence_features=x.new_zeros((batch_size, n_var, hidden_dim)),
            global_features=x.new_zeros((batch_size, hidden_dim)),
            sequence_mask=variable_mask,
        )

    def _run_optional_irregular_branch(self, aligned_batch: BranchAlignedBatch) -> BranchFeatures:
        if self.use_irregular_branch:
            return self._run_irregular_branch(aligned_batch)
        return self._null_branch_features(
            aligned_batch.values,
            aligned_batch.mask,
            self.config.irregular_hidden_dim,
        )

    def _encode_future_queries(self, y_mark: Tensor) -> Tensor | None:
        """把未来待预测时间编码成 query 向量。"""
        if not self.use_time_features:
            return None
        return self.time_encoder(y_mark)

    def _reduce_foundation_features(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> BranchFeatures:
        """在进入 fusion 之前，对 foundation 路做条件相关的粗筛与降维。"""
        c2lite_mode = getattr(self.foundation_branch, "scale_mode", "none") == "expert_routing_c2lite"
        return self.foundation_reducer(
            foundation_features=foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
            use_irregular_context=not c2lite_mode,
            use_future_context=True,
            disable_gate=c2lite_mode,
        )

    def _fuse_branch_features(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> Tensor:
        return self.fusion_router(
            foundation_features=foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )

    def _predict(self, fused_features: Tensor) -> Tensor:
        return self.prediction_head(fused_features)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor,
        x_mask: Tensor,
        **kwargs,
    ) -> dict[str, Tensor | None]:
        """完整前向：
        输入历史观测和未来时间，输出预测值、真值与 mask。
        """
        y_mark: Tensor = kwargs["y_mark"]
        y: Tensor | None = kwargs.get("y")
        y_mask: Tensor | None = kwargs.get("y_mask")

        self._validate_inputs(x=x, x_mark=x_mark, x_mask=x_mask, y_mark=y_mark)

        aligned_batch = self._align_structure(x=x, x_mark=x_mark, x_mask=x_mask)
        future_queries = self._encode_future_queries(y_mark=y_mark)

        irregular_features = self._run_optional_irregular_branch(aligned_batch.irregular)
        if self.use_foundation_branch:
            if self.foundation_branch is None:
                raise RuntimeError("Foundation branch state is inconsistent.")
            if self.foundation_branch.scale_mode == "none":
                foundation_features = self._run_foundation_branch(aligned_batch.foundation)
            else:
                scale_context = ScaleRoutingContext(
                    future_queries=future_queries,
                    irregular_global_features=(
                        irregular_features.global_features if self.use_irregular_branch else None
                    ),
                )
                foundation_features = self._run_foundation_branch(
                    aligned_batch.foundation,
                    scale_context=scale_context,
                )
            reduced_foundation_features = self._reduce_foundation_features(
                foundation_features=foundation_features,
                irregular_features=irregular_features,
                future_queries=future_queries,
            )
        else:
            reduced_foundation_features = self._null_branch_features(
                x,
                x_mask,
                self.config.foundation_reduced_dim,
            )
        fused_features = self._fuse_branch_features(
            foundation_features=reduced_foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )
        predictions = self._predict(fused_features=fused_features)

        f_dim = -1 if self.features == "MS" else 0
        pred = predictions[:, :, f_dim:]
        true = y[:, :, f_dim:] if y is not None else None
        mask = y_mask[:, :, f_dim:] if y_mask is not None else None

        return {
            "pred": pred,
            "true": true,
            "mask": mask,
        }
