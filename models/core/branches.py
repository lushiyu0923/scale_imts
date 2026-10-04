from __future__ import annotations

"""ScaleIMTS 的两条分支、融合器和输出头。"""
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from .backbones import build_bert_backbone, build_gpt_backbone

from .config import BranchAlignedBatch, BranchFeatures, ScaleIMTSConfig, ScaleRoutingContext
from .irregular_aligners import (
    DynamicGraphBlock,
    PatchIntraEncoder,
    PatchTemporalEncoder,
)
from .scale_routing import (
    ScaleExpertBank,
    ScaleRouter,
    ScaleSegmentExpertBank,
    ScaleStatsEncoder,
    routing_entropy,
    summarize_router_usage,
)
from .time_encoders import (
    TimeFeatureEncoder,
)


def _zero_init_last_linear(module: nn.Sequential) -> None:
    """Initialize the last linear layer to zero so the block starts near identity."""
    last_linear = module[-1]
    if not isinstance(last_linear, nn.Linear):
        raise TypeError("Expected the last module to be nn.Linear.")
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)


class FoundationBranch(nn.Module):
    """ISTS-PLM-inspired two-stage PLM branch.

    Stage 1: intra-series / temporal modeling on per-variable token sequences
    Stage 2: inter-series / variable modeling on per-variable pooled features
    """

    def __init__(self, enc_in: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.enc_in = enc_in
        self.hidden_dim = hidden_dim
        self.ln_proj = nn.LayerNorm(hidden_dim)
        self.feature_dropout = nn.Dropout(dropout)
        self.temporal_backbone = None
        self.variable_backbone = None
        self.semi_freeze = False
        self.train_last_n_layers = 0
        self.ablation_variant = "full"
        self.use_local_scale_adaptation = True

        self.scale_mode = "none"
        self.scale_num_segments = 1
        self.scale_num_experts = 1
        self.scale_temperature = 1.0
        self.scale_use_future_context = True
        self.scale_use_irregular_context = True
        self.scale_irregular_proj: nn.Linear | None = None
        self.scale_stats_encoder: ScaleStatsEncoder | None = None
        self.scale_router: ScaleRouter | None = None
        self.scale_experts: ScaleExpertBank | None = None
        self.scale_segment_experts: ScaleSegmentExpertBank | None = None
        self.scale_summary_proj: nn.Sequential | None = None
        self.scale_residual_alpha_logit: nn.Parameter | None = None
        self.scale_residual_alpha_max: float = 0.3
        self.last_scale_router_weights: Tensor | None = None
        self.last_scale_router_logits: Tensor | None = None
        self.last_scale_router_entropy: Tensor | None = None
        self.last_scale_router_usage: Tensor | None = None

    @staticmethod
    def _build_gpt_backbone(hidden_dim: int, n_layers: int, dropout: float, pretrained_path: Path):
        return build_gpt_backbone(hidden_dim=hidden_dim, n_layers=n_layers, pretrained_path=pretrained_path)

    @staticmethod
    def _build_bert_backbone(hidden_dim: int, n_layers: int, dropout: float, pretrained_path: Path):
        return build_bert_backbone(hidden_dim=hidden_dim, n_layers=n_layers, pretrained_path=pretrained_path)

    @staticmethod
    def _get_transformer_layers(backbone: nn.Module) -> list[nn.Module]:
        if hasattr(backbone, "h"):
            return list(backbone.h)
        encoder = getattr(backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            return list(encoder.layer)
        raise TypeError("Unsupported backbone structure for partial fine-tuning.")

    def _apply_backbone_train_policy(self, backbone: nn.Module) -> None:
        for param in backbone.parameters():
            param.requires_grad = False

        if self.train_last_n_layers > 0:
            layers = self._get_transformer_layers(backbone)
            keep_n = min(self.train_last_n_layers, len(layers))
            for layer in layers[-keep_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        if self.semi_freeze or self.train_last_n_layers > 0:
            for name, param in backbone.named_parameters():
                if "ln" in name.lower() or "layernorm" in name.lower() or "norm" in name.lower():
                    param.requires_grad = True

    @classmethod
    def from_config(cls, config: ScaleIMTSConfig):
        """Build the foundation branch from config and load local wope weights."""
        branch = cls(
            enc_in=config.enc_in,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        branch.scale_mode = (config.foundation_scale_mode or "none").lower()
        branch.ablation_variant = (config.ablation_variant or "full").lower()
        if branch.ablation_variant not in {
            "full",
            "irregular_only",
            "wo_local_scale",
            "wo_irregular_dynamics",
            "wo_foundation_refinement",
            "concat_fusion",
            "global_routing",
            "wo_scale_statistics",
        }:
            raise ValueError(f"Unsupported ScaleIMTS ablation_variant: {config.ablation_variant}")
        branch.use_local_scale_adaptation = branch.ablation_variant != "wo_local_scale"
        if branch.scale_mode not in {"none", "expert_routing", "expert_routing_c2lite"}:
            raise ValueError(f"Unsupported foundation_scale_mode: {config.foundation_scale_mode}")
        branch.scale_num_segments = max(1, int(config.foundation_scale_num_segments or 1))
        if branch.ablation_variant == "global_routing":
            branch.scale_num_segments = 1
        branch.scale_num_experts = max(1, int(config.foundation_scale_num_experts or 1))
        branch.scale_temperature = float(config.foundation_scale_temperature or 1.0)
        branch.scale_use_future_context = (
            config.foundation_scale_use_future_context
            if config.foundation_scale_use_future_context is not None
            else True
        )
        branch.scale_use_irregular_context = (
            config.foundation_scale_use_irregular_context
            if config.foundation_scale_use_irregular_context is not None
            else True
        )
        branch.train_last_n_layers = max(0, int(config.foundation_train_last_n_layers or 0))
        branch.semi_freeze = bool(config.foundation_semi_freeze)

        te_model = config.foundation_te_model.lower()
        st_model = config.foundation_st_model.lower()
        pretrained_root = Path(config.foundation_pretrained_root)
        gpt_path = pretrained_root / "gpt2"
        bert_path = pretrained_root / "bert-base-uncased"

        if te_model == "gpt":
            branch.temporal_backbone = branch._build_gpt_backbone(
                hidden_dim=config.hidden_dim,
                n_layers=config.foundation_n_te_layers,
                dropout=config.dropout,
                pretrained_path=gpt_path,
            )
        elif te_model == "bert":
            branch.temporal_backbone = branch._build_bert_backbone(
                hidden_dim=config.hidden_dim,
                n_layers=config.foundation_n_te_layers,
                dropout=config.dropout,
                pretrained_path=bert_path,
            )
        else:
            raise ValueError(f"Unsupported foundation_te_model: {config.foundation_te_model}")

        if st_model == "gpt":
            branch.variable_backbone = branch._build_gpt_backbone(
                hidden_dim=config.hidden_dim,
                n_layers=config.foundation_n_st_layers,
                dropout=config.dropout,
                pretrained_path=gpt_path,
            )
        elif st_model == "bert":
            branch.variable_backbone = branch._build_bert_backbone(
                hidden_dim=config.hidden_dim,
                n_layers=config.foundation_n_st_layers,
                dropout=config.dropout,
                pretrained_path=bert_path,
            )
        else:
            raise ValueError(f"Unsupported foundation_st_model: {config.foundation_st_model}")

        branch._apply_backbone_train_policy(branch.temporal_backbone)
        branch._apply_backbone_train_policy(branch.variable_backbone)

        if branch.scale_mode != "none":
            branch.scale_irregular_proj = nn.Linear(config.irregular_hidden_dim, config.hidden_dim)
            branch.scale_stats_encoder = ScaleStatsEncoder(
                hidden_dim=config.hidden_dim,
                num_segments=branch.scale_num_segments,
                dropout=config.dropout,
                use_explicit_statistics=branch.ablation_variant != "wo_scale_statistics",
            )
            branch.scale_router = ScaleRouter(
                hidden_dim=config.hidden_dim,
                num_experts=branch.scale_num_experts,
                router_hidden_dim=config.foundation_scale_router_hidden_dim,
                temperature=branch.scale_temperature,
                dropout=config.dropout,
            )
            if branch.scale_mode == "expert_routing":
                branch.scale_experts = ScaleExpertBank(
                    hidden_dim=config.hidden_dim,
                    num_experts=branch.scale_num_experts,
                    expert_hidden_dim=config.foundation_scale_expert_hidden_dim,
                    dropout=config.dropout,
                )
                branch.scale_summary_proj = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                )
                _zero_init_last_linear(branch.scale_summary_proj)
            elif branch.scale_mode == "expert_routing_c2lite":
                if branch.scale_num_experts != 3:
                    raise ValueError("expert_routing_c2lite currently expects foundation_scale_num_experts=3.")
                branch.scale_segment_experts = ScaleSegmentExpertBank(
                    hidden_dim=config.hidden_dim,
                    num_experts=branch.scale_num_experts,
                    expert_hidden_dim=config.foundation_scale_expert_hidden_dim,
                    dropout=config.dropout,
                )
                branch.scale_residual_alpha_logit = nn.Parameter(torch.tensor(-4.0))

        return branch

    @staticmethod
    def _forward_backbone(backbone: nn.Module, inputs_embeds: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _run_scale_routing(
        self,
        aligned_batch: BranchAlignedBatch,
        temporal_outputs: Tensor,
        token_mask: Tensor,
        scale_context: ScaleRoutingContext | None,
    ) -> Tensor | None:
        if self.scale_mode == "none":
            return None
        if self.scale_stats_encoder is None or self.scale_router is None or self.scale_experts is None:
            raise RuntimeError("Scale routing modules have not been initialized.")
        if self.scale_summary_proj is None or self.scale_irregular_proj is None:
            raise RuntimeError("Scale routing projections have not been initialized.")

        batch_size, _, n_var = aligned_batch.values.shape
        seq_len = temporal_outputs.shape[1] - 1
        core_temporal = temporal_outputs[:, 1:, :]
        core_mask = token_mask[:, 1:, :]
        if aligned_batch.time is None:
            time_values = None
        else:
            time_values = aligned_batch.time.repeat(1, 1, n_var)
            time_values = time_values.permute(0, 2, 1).contiguous().reshape(batch_size * n_var, seq_len, 1)

        segment_descriptor, segment_mask, _ = self.scale_stats_encoder(
            sequence_features=core_temporal,
            token_mask=core_mask,
            time_values=time_values,
        )
        num_segments = segment_descriptor.shape[1]

        if aligned_batch.variable_prompts is not None:
            variable_context = aligned_batch.variable_prompts.squeeze(1).contiguous().reshape(batch_size * n_var, 1, self.hidden_dim)
            variable_context = variable_context.expand(-1, num_segments, -1)
        else:
            variable_context = torch.zeros_like(segment_descriptor)

        future_context = None
        if self.scale_use_future_context and scale_context is not None and scale_context.future_queries is not None:
            future_summary = scale_context.future_queries.mean(dim=1)
            future_context = future_summary.unsqueeze(1).unsqueeze(1).expand(
                batch_size,
                n_var,
                num_segments,
                -1,
            ).contiguous().reshape(batch_size * n_var, num_segments, self.hidden_dim)

        irregular_context = None
        if self.scale_use_irregular_context and scale_context is not None and scale_context.irregular_global_features is not None:
            irregular_summary = self.scale_irregular_proj(scale_context.irregular_global_features)
            irregular_context = irregular_summary.unsqueeze(1).unsqueeze(1).expand(
                batch_size,
                n_var,
                num_segments,
                -1,
            ).contiguous().reshape(batch_size * n_var, num_segments, self.hidden_dim)

        router_weights, router_logits = self.scale_router(
            segment_descriptor=segment_descriptor,
            segment_mask=segment_mask,
            future_context=future_context,
            irregular_context=irregular_context,
            variable_context=variable_context,
        )
        mixed_segments, _ = self.scale_experts(
            segment_descriptor=segment_descriptor,
            router_weights=router_weights,
        )

        segment_mask_expanded = segment_mask.unsqueeze(-1).float()
        segment_denom = segment_mask_expanded.sum(dim=1).clamp_min(1e-8)
        scale_summary = (mixed_segments * segment_mask_expanded).sum(dim=1) / segment_denom
        scale_delta = self.scale_summary_proj(scale_summary).view(batch_size, n_var, self.hidden_dim)

        self.last_scale_router_weights = router_weights.detach()
        self.last_scale_router_logits = router_logits.detach()
        self.last_scale_router_entropy = routing_entropy(router_weights).mean().detach()
        self.last_scale_router_usage = summarize_router_usage(router_weights).detach()
        return scale_delta

    def _run_scale_routing_c2lite(
        self,
        aligned_batch: BranchAlignedBatch,
        temporal_outputs: Tensor,
        token_mask: Tensor,
        scale_context: ScaleRoutingContext | None,
    ) -> Tensor | None:
        if self.scale_mode != "expert_routing_c2lite":
            return None
        if self.scale_stats_encoder is None or self.scale_router is None or self.scale_segment_experts is None:
            raise RuntimeError("C2-lite scale routing modules have not been initialized.")
        if self.scale_irregular_proj is None:
            raise RuntimeError("Scale routing projections have not been initialized.")
        if self.scale_residual_alpha_logit is None:
            raise RuntimeError("Scale residual alpha has not been initialized.")

        batch_size, _, n_var = aligned_batch.values.shape
        seq_len = temporal_outputs.shape[1] - 1
        core_temporal = temporal_outputs[:, 1:, :]
        core_mask = token_mask[:, 1:, :].squeeze(-1)

        if aligned_batch.time is None:
            time_values = None
        else:
            time_values = aligned_batch.time.repeat(1, 1, n_var)
            time_values = time_values.permute(0, 2, 1).contiguous().reshape(batch_size * n_var, seq_len, 1)

        core_temporal, core_mask, time_values, segment_len = self.scale_stats_encoder._pad_to_segments(  # type: ignore[attr-defined]
            sequence_features=core_temporal,
            token_mask=core_mask,
            time_values=time_values,
        )
        padded_len = core_temporal.shape[1]
        num_segments = padded_len // segment_len

        segment_descriptor, segment_mask, segment_stats = self.scale_stats_encoder(
            sequence_features=core_temporal,
            token_mask=core_mask,
            time_values=time_values,
        )

        segment_tokens = core_temporal.contiguous().view(batch_size * n_var, num_segments, segment_len, self.hidden_dim)
        segment_token_mask = core_mask.contiguous().view(batch_size * n_var, num_segments, segment_len).bool()

        if aligned_batch.variable_prompts is not None:
            variable_context = aligned_batch.variable_prompts.squeeze(1).contiguous().reshape(batch_size * n_var, 1, self.hidden_dim)
            variable_context = variable_context.expand(-1, num_segments, -1)
        else:
            variable_context = torch.zeros_like(segment_descriptor)

        future_context = None
        if self.scale_use_future_context and scale_context is not None and scale_context.future_queries is not None:
            future_summary = scale_context.future_queries.mean(dim=1)
            future_context = future_summary.unsqueeze(1).unsqueeze(1).expand(
                batch_size,
                n_var,
                num_segments,
                -1,
            ).contiguous().reshape(batch_size * n_var, num_segments, self.hidden_dim)

        irregular_context = None
        if self.scale_use_irregular_context and scale_context is not None and scale_context.irregular_global_features is not None:
            irregular_summary = self.scale_irregular_proj(scale_context.irregular_global_features)
            irregular_context = irregular_summary.unsqueeze(1).unsqueeze(1).expand(
                batch_size,
                n_var,
                num_segments,
                -1,
            ).contiguous().reshape(batch_size * n_var, num_segments, self.hidden_dim)

        router_weights, router_logits = self.scale_router(
            segment_descriptor=segment_descriptor,
            segment_mask=segment_mask,
            future_context=future_context,
            irregular_context=irregular_context,
            variable_context=variable_context,
        )
        mixed_segments, _ = self.scale_segment_experts(
            segment_tokens=segment_tokens,
            segment_mask=segment_token_mask,
            segment_descriptor=segment_descriptor,
            segment_stats=segment_stats,
            router_weights=router_weights,
        )

        alpha = torch.sigmoid(self.scale_residual_alpha_logit) * self.scale_residual_alpha_max
        segment_delta_tokens = mixed_segments.unsqueeze(2).expand(-1, -1, segment_len, -1).reshape(
            batch_size * n_var,
            padded_len,
            self.hidden_dim,
        )
        segment_delta_tokens = segment_delta_tokens * core_mask.unsqueeze(-1).float()
        core_temporal = core_temporal + alpha * segment_delta_tokens
        core_temporal = core_temporal[:, :seq_len, :]

        self.last_scale_router_weights = router_weights.detach()
        self.last_scale_router_logits = router_logits.detach()
        self.last_scale_router_entropy = routing_entropy(router_weights).mean().detach()
        self.last_scale_router_usage = summarize_router_usage(router_weights).detach()

        return torch.cat([temporal_outputs[:, :1, :], core_temporal], dim=1)

    def forward(
        self,
        aligned_batch: BranchAlignedBatch,
        scale_context: ScaleRoutingContext | None = None,
    ) -> BranchFeatures:
        """Foundation forward path.

        1. temporal backbone encodes per-variable token sequences;
        2. tokens are pooled into variable-level features;
        3. optional scale routing adapts the variable features;
        4. variable backbone models inter-series relations;
        5. variable features are pooled into a global summary.
        """
        if aligned_batch.tokens is None or aligned_batch.token_mask is None:
            raise ValueError("FoundationBranch expects tokenized outputs from FoundationAligner.")
        if self.temporal_backbone is None or self.variable_backbone is None:
            raise RuntimeError("FoundationBranch backbones have not been initialized.")

        self.last_scale_router_weights = None
        self.last_scale_router_logits = None
        self.last_scale_router_entropy = None
        self.last_scale_router_usage = None

        batch_size, _, n_var = aligned_batch.values.shape
        token_mask = aligned_batch.token_mask.squeeze(-1)
        token_mask_expanded = aligned_batch.token_mask

        temporal_outputs = self._forward_backbone(
            backbone=self.temporal_backbone,
            inputs_embeds=aligned_batch.tokens,
            attention_mask=token_mask,
        )

        if self.scale_mode == "expert_routing_c2lite" and self.use_local_scale_adaptation:
            temporal_outputs = self._run_scale_routing_c2lite(
                aligned_batch=aligned_batch,
                temporal_outputs=temporal_outputs,
                token_mask=token_mask_expanded,
                scale_context=scale_context,
            )
        n_nonmask = token_mask_expanded.sum(dim=1).clamp_min(1e-8)
        pooled = (temporal_outputs * token_mask_expanded).sum(dim=1) / n_nonmask
        sequence_features = pooled.view(batch_size, n_var, self.hidden_dim)

        if self.scale_mode == "expert_routing":
            scale_delta = self._run_scale_routing(
                aligned_batch=aligned_batch,
                temporal_outputs=temporal_outputs,
                token_mask=token_mask_expanded,
                scale_context=scale_context,
            )
            if scale_delta is not None:
                sequence_features = sequence_features + scale_delta

        sequence_features = self.ln_proj(sequence_features)
        sequence_features = self.feature_dropout(sequence_features)

        if aligned_batch.variable_prompts is not None:
            sequence_features = sequence_features + aligned_batch.variable_prompts.squeeze(1)
            sequence_features = self.feature_dropout(sequence_features)

        variable_mask = (aligned_batch.mask.sum(dim=1) > 0).float()
        variable_outputs = self._forward_backbone(
            backbone=self.variable_backbone,
            inputs_embeds=sequence_features,
            attention_mask=variable_mask,
        )
        variable_outputs = self.feature_dropout(variable_outputs)

        variable_mask_expanded = variable_mask.unsqueeze(-1)
        n_valid_vars = variable_mask_expanded.sum(dim=1).clamp_min(1e-8)
        global_features = (variable_outputs * variable_mask_expanded).sum(dim=1) / n_valid_vars

        return BranchFeatures(
            sequence_features=variable_outputs,
            global_features=global_features,
            sequence_mask=variable_mask.bool(),
            routing_weights=self.last_scale_router_weights,
        )


class FoundationReducer(nn.Module):
    """foundation 路降维模块：先粗筛，再压缩。

    这里实现的是当前阶段的过渡版本：
    1. 用 foundation / irregular / future query 的全局条件，对每个变量 token 做软门控；
    2. 再把筛过的 768 维 foundation 表示压缩到较小的 reduced_dim；
    3. 用 gated weighted pooling 得到压缩后的全局摘要。
    """

    def __init__(
        self,
        foundation_hidden_dim: int,
        irregular_hidden_dim: int,
        reduced_dim: int,
        gate_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.foundation_hidden_dim = foundation_hidden_dim
        self.irregular_hidden_dim = irregular_hidden_dim
        self.reduced_dim = reduced_dim

        self.gate_irregular_proj = nn.Linear(irregular_hidden_dim, foundation_hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(foundation_hidden_dim * 4, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )

        # 这是当前阶段的临时占位压缩器：先保证能跑通 foundation 降维链路，
        # 后续可以替换成更强的 latent readout / cross-attention 压缩。
        self.sequence_reducer = nn.Sequential(
            nn.Linear(foundation_hidden_dim, reduced_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_residual_proj = nn.Linear(foundation_hidden_dim, reduced_dim)
        self.sequence_norm = nn.LayerNorm(reduced_dim)
        self.global_norm = nn.LayerNorm(reduced_dim)
        self._init_stable_parameters()

    def _init_stable_parameters(self) -> None:
        """Keep the routing gate close to a neutral 0.5 state at startup."""
        gate_head = self.gate_mlp[-1]
        if not isinstance(gate_head, nn.Linear):
            raise TypeError("Expected gate_mlp to end with nn.Linear.")
        nn.init.zeros_(gate_head.weight)
        nn.init.zeros_(gate_head.bias)

    def forward(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
        *,
        use_irregular_context: bool = True,
        use_future_context: bool = True,
        disable_gate: bool = False,
    ) -> BranchFeatures:
        foundation_sequence = foundation_features.sequence_features
        foundation_global = foundation_features.global_features
        irregular_global = irregular_features.global_features if use_irregular_context else torch.zeros_like(foundation_global)

        batch_size, n_var, _ = foundation_sequence.shape
        if future_queries is None or not use_future_context:
            q_summary = torch.zeros_like(foundation_global)
        else:
            q_summary = future_queries.mean(dim=1)

        if disable_gate:
            reduced_sequence = self.sequence_norm(self.sequence_reducer(foundation_sequence))
            sequence_mask = foundation_features.sequence_mask
            if sequence_mask is not None:
                valid_mask = sequence_mask.unsqueeze(-1).float()
                n_valid = valid_mask.sum(dim=1).clamp_min(1e-8)
                reduced_global = (reduced_sequence * valid_mask).sum(dim=1) / n_valid
            else:
                reduced_global = reduced_sequence.mean(dim=1)
            reduced_global = self.global_norm(reduced_global + self.global_residual_proj(foundation_global))
            return BranchFeatures(
                sequence_features=reduced_sequence,
                global_features=reduced_global,
                sequence_mask=foundation_features.sequence_mask,
                routing_weights=None,
            )

        foundation_global_expand = foundation_global.unsqueeze(1).expand(-1, n_var, -1)
        q_summary_expand = q_summary.unsqueeze(1).expand(-1, n_var, -1)
        irregular_global_proj = self.gate_irregular_proj(irregular_global)
        irregular_global_expand = irregular_global_proj.unsqueeze(1).expand(-1, n_var, -1)

        gate_input = torch.cat(
            [
                foundation_sequence,
                foundation_global_expand,
                irregular_global_expand,
                q_summary_expand,
            ],
            dim=-1,
        )
        gate_logits = self.gate_mlp(gate_input)
        if foundation_features.sequence_mask is not None:
            gate_logits = gate_logits.masked_fill(~foundation_features.sequence_mask.unsqueeze(-1), -1e8)
        gate = torch.sigmoid(gate_logits)

        filtered_sequence = gate * foundation_sequence
        reduced_sequence = self.sequence_norm(self.sequence_reducer(filtered_sequence))

        gate_weights = gate
        if foundation_features.sequence_mask is not None:
            gate_weights = gate_weights * foundation_features.sequence_mask.unsqueeze(-1).float()
        n_valid = gate_weights.sum(dim=1).clamp_min(1e-8)
        reduced_global = (reduced_sequence * gate_weights).sum(dim=1) / n_valid
        reduced_global = self.global_norm(reduced_global + self.global_residual_proj(foundation_global))

        return BranchFeatures(
            sequence_features=reduced_sequence,
            global_features=reduced_global,
            sequence_mask=foundation_features.sequence_mask,
            routing_weights=gate,
        )


class IrregularBranch(nn.Module):
    """Irregular branch inspired by t-PatchGNN's problem decomposition.

    The branch is split into three stages:
    1. patch-intra irregular aggregation
    2. patch-inter temporal modeling
    3. time-adaptive cross-variable graph interaction
    """

    def __init__(
        self,
        enc_in: int,
        hidden_dim: int,
        irregular_hidden_dim: int,
        irregular_time_dim: int,
        irregular_node_dim: int,
        patch_len: int,
        max_time_steps: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        del enc_in, patch_len, max_time_steps
        self.irregular_hidden_dim = irregular_hidden_dim
        self.patch_intra_encoder = PatchIntraEncoder(
            hidden_dim=irregular_hidden_dim,
            time_dim=irregular_time_dim,
            dropout=dropout,
        )
        self.patch_temporal_encoder = PatchTemporalEncoder(
            hidden_dim=irregular_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dynamic_graph_block = DynamicGraphBlock(
            hidden_dim=irregular_hidden_dim,
            node_dim=irregular_node_dim,
            dropout=dropout,
        )
        self.sequence_norm = nn.LayerNorm(irregular_hidden_dim)
        self.global_norm = nn.LayerNorm(irregular_hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, aligned_batch: BranchAlignedBatch) -> BranchFeatures:
        """irregular 路前向：
        1. patch 内聚合；
        2. patch 间时间建模；
        3. 动态图传播；
        4. patch 维池化；
        5. 变量维池化；
        6. 投影到与 foundation 路对齐的输出空间。
        """
        if (
            aligned_batch.patch_values is None
            or aligned_batch.patch_times is None
            or aligned_batch.patch_masks is None
            or aligned_batch.patch_valid is None
        ):
            raise ValueError("IrregularBranch expects patch-level outputs from IrregularAligner.")

        patch_repr = self.patch_intra_encoder(
            patch_values=aligned_batch.patch_values,
            patch_times=aligned_batch.patch_times,
            patch_masks=aligned_batch.patch_masks,
        )

        temporal_repr = self.patch_temporal_encoder(
            patch_repr=patch_repr,
            patch_valid=aligned_batch.patch_valid,
        )

        graph_repr = self.dynamic_graph_block(
            patch_repr=temporal_repr,
            patch_valid=aligned_batch.patch_valid,
        )

        patch_valid = aligned_batch.patch_valid
        # irregular 路第一处 masked mean pooling：把多个 patch 的表示压成每变量表示。
        n_valid_patches = patch_valid.sum(dim=2).clamp_min(1e-8)
        sequence_features_small = (graph_repr * patch_valid).sum(dim=2) / n_valid_patches
        sequence_features_small = self.output_dropout(sequence_features_small)

        variable_valid = (aligned_batch.mask.sum(dim=1) > 0).float().unsqueeze(-1)
        # irregular 路第二处 masked mean pooling：把变量级表示压成全局表示。
        n_valid_vars = variable_valid.sum(dim=1).clamp_min(1e-8)
        global_features_small = (sequence_features_small * variable_valid).sum(dim=1) / n_valid_vars
        global_features_small = self.output_dropout(global_features_small)

        # irregular 路保持自己的原生表征维度，不在分支尾部强行拉到 foundation 的 768 维。
        sequence_features = self.sequence_norm(sequence_features_small)
        global_features = self.global_norm(global_features_small)

        return BranchFeatures(
            sequence_features=sequence_features,
            global_features=global_features,
            sequence_mask=variable_valid.squeeze(-1).bool(),
        )


class FusionRouter(nn.Module):
    """Fusion module for branch features.

    - concat: global feature concatenation baseline
    - film_f2i: foundation-conditioned FiLM modulation for irregular features
    - film_i2f: irregular-conditioned FiLM modulation for foundation features
    - film_f2i_var: foundation-conditioned variable-level FiLM for irregular features
    - film_i2f_var: irregular-conditioned variable-level FiLM for foundation features
    """

    def __init__(
        self,
        hidden_dim: int,
        foundation_reduced_dim: int,
        irregular_hidden_dim: int,
        fusion_type: str,
        fusion_hidden_dim: int,
        film_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.foundation_reduced_dim = foundation_reduced_dim
        self.irregular_hidden_dim = irregular_hidden_dim
        self.fusion_type = fusion_type.lower()
        self.fusion_hidden_dim = fusion_hidden_dim

        if self.fusion_type not in {
            "concat",
            "film_f2i",
            "film_i2f",
            "film_f2i_var",
            "film_i2f_var",
            "add_i2f_var",
        }:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.concat_foundation_proj = (
            nn.Identity() if foundation_reduced_dim == hidden_dim else nn.Linear(foundation_reduced_dim, hidden_dim)
        )
        self.concat_irregular_proj = (
            nn.Identity() if irregular_hidden_dim == hidden_dim else nn.Linear(irregular_hidden_dim, hidden_dim)
        )
        self.fusion_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.concat_output_proj = nn.Linear(hidden_dim, fusion_hidden_dim)

        self.foundation_small_proj = nn.Linear(foundation_reduced_dim, fusion_hidden_dim)
        self.irregular_small_proj = nn.Linear(irregular_hidden_dim, fusion_hidden_dim)
        self.query_small_proj = nn.Linear(hidden_dim, fusion_hidden_dim)
        self.foundation_norm = nn.LayerNorm(fusion_hidden_dim)
        self.irregular_norm = nn.LayerNorm(fusion_hidden_dim)
        self.query_norm = nn.LayerNorm(fusion_hidden_dim)
        self.fused_norm = nn.LayerNorm(fusion_hidden_dim)

        self.foundation_to_irregular = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, film_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(film_hidden_dim, fusion_hidden_dim * 2),
        )
        self.irregular_to_foundation = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, film_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(film_hidden_dim, fusion_hidden_dim * 2),
        )
        self.foundation_to_irregular_var = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, film_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(film_hidden_dim, fusion_hidden_dim * 2),
        )
        self.irregular_to_foundation_var = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, film_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(film_hidden_dim, fusion_hidden_dim * 2),
        )
        self._init_stable_parameters()

    def _init_stable_parameters(self) -> None:
        """Make FiLM heads start as identity modulation."""
        for module in (
            self.foundation_to_irregular,
            self.irregular_to_foundation,
            self.foundation_to_irregular_var,
            self.irregular_to_foundation_var,
        ):
            _zero_init_last_linear(module)

    def _forward_concat(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> Tensor:
        batch_size = foundation_features.global_features.shape[0]
        pred_len = future_queries.shape[1] if future_queries is not None else 1

        foundation_context = foundation_features.global_features.unsqueeze(1).expand(-1, pred_len, -1)
        foundation_context = self.concat_foundation_proj(foundation_context)
        irregular_context = irregular_features.global_features.unsqueeze(1).expand(-1, pred_len, -1)
        irregular_context = self.concat_irregular_proj(irregular_context)

        if future_queries is None:
            future_queries = torch.zeros(
                (batch_size, pred_len, self.hidden_dim),
                dtype=foundation_context.dtype,
                device=foundation_context.device,
            )

        fused = torch.cat([foundation_context, irregular_context, future_queries], dim=-1)
        fused = self.fusion_dropout(fused)
        return self.concat_output_proj(self.output_proj(fused))

    def _prepare_small_contexts(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = foundation_features.global_features.shape[0]
        pred_len = future_queries.shape[1] if future_queries is not None else 1

        foundation_context = foundation_features.global_features.unsqueeze(1).expand(-1, pred_len, -1)
        irregular_context = irregular_features.global_features.unsqueeze(1).expand(-1, pred_len, -1)

        if future_queries is None:
            future_queries = torch.zeros(
                (batch_size, pred_len, self.hidden_dim),
                dtype=foundation_context.dtype,
                device=foundation_context.device,
            )

        foundation_context = self.foundation_norm(self.foundation_small_proj(foundation_context))
        irregular_context = self.irregular_norm(self.irregular_small_proj(irregular_context))
        query_context = self.query_norm(self.query_small_proj(future_queries))
        return foundation_context, irregular_context, query_context

    def _prepare_small_sequence_contexts(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, n_var, _ = foundation_features.sequence_features.shape
        pred_len = future_queries.shape[1] if future_queries is not None else 1

        foundation_sequence = self.foundation_norm(self.foundation_small_proj(foundation_features.sequence_features))
        irregular_sequence = self.irregular_norm(self.irregular_small_proj(irregular_features.sequence_features))

        if future_queries is None:
            future_queries = torch.zeros(
                (batch_size, pred_len, self.hidden_dim),
                dtype=foundation_sequence.dtype,
                device=foundation_sequence.device,
            )

        query_context = self.query_norm(self.query_small_proj(future_queries))
        foundation_context = foundation_sequence.unsqueeze(1).expand(-1, pred_len, -1, -1)
        irregular_context = irregular_sequence.unsqueeze(1).expand(-1, pred_len, -1, -1)
        query_context = query_context.unsqueeze(2).expand(-1, -1, n_var, -1)
        return foundation_context, irregular_context, query_context

    @staticmethod
    def _apply_film(modulator_out: Tensor, target: Tensor) -> Tensor:
        gamma, beta = torch.chunk(modulator_out, chunks=2, dim=-1)
        gamma = 1.0 + gamma
        return gamma * target + beta

    def _forward_film_f2i(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> Tensor:
        foundation_context, irregular_context, query_context = self._prepare_small_contexts(
            foundation_features=foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )
        modulator_input = torch.cat([foundation_context, query_context], dim=-1)
        modulator_out = self.foundation_to_irregular(modulator_input)
        irregular_modulated = self._apply_film(modulator_out, irregular_context)
        fused = self.fused_norm(irregular_modulated + query_context)
        return fused

    def _forward_film_i2f(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> Tensor:
        foundation_context, irregular_context, query_context = self._prepare_small_contexts(
            foundation_features=foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )
        modulator_input = torch.cat([irregular_context, query_context], dim=-1)
        modulator_out = self.irregular_to_foundation(modulator_input)
        foundation_modulated = self._apply_film(modulator_out, foundation_context)
        fused = self.fused_norm(foundation_modulated + query_context)
        return fused

    def _forward_film_f2i_var(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> Tensor:
        foundation_context, irregular_context, query_context = self._prepare_small_sequence_contexts(
            foundation_features=foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )
        modulator_input = torch.cat([foundation_context, query_context], dim=-1)
        modulator_out = self.foundation_to_irregular_var(modulator_input)
        irregular_modulated = self._apply_film(modulator_out, irregular_context)
        return self.fused_norm(irregular_modulated + query_context)

    def _forward_film_i2f_var(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> Tensor:
        foundation_context, irregular_context, query_context = self._prepare_small_sequence_contexts(
            foundation_features=foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )
        modulator_input = torch.cat([irregular_context, query_context], dim=-1)
        modulator_out = self.irregular_to_foundation_var(modulator_input)
        foundation_modulated = self._apply_film(modulator_out, foundation_context)
        return self.fused_norm(foundation_modulated + query_context)

    def _forward_add_i2f_var(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> Tensor:
        foundation_context, irregular_context, query_context = self._prepare_small_sequence_contexts(
            foundation_features=foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )
        return self.fused_norm(foundation_context + irregular_context + query_context)

    def forward(
        self,
        foundation_features: BranchFeatures,
        irregular_features: BranchFeatures,
        future_queries: Tensor | None,
    ) -> Tensor:
        """Route fusion according to ``fusion_type``."""
        if self.fusion_type == "concat":
            return self._forward_concat(
                foundation_features=foundation_features,
                irregular_features=irregular_features,
                future_queries=future_queries,
            )
        if self.fusion_type == "film_f2i":
            return self._forward_film_f2i(
                foundation_features=foundation_features,
                irregular_features=irregular_features,
                future_queries=future_queries,
            )
        if self.fusion_type == "film_f2i_var":
            return self._forward_film_f2i_var(
                foundation_features=foundation_features,
                irregular_features=irregular_features,
                future_queries=future_queries,
            )
        if self.fusion_type == "film_i2f_var":
            return self._forward_film_i2f_var(
                foundation_features=foundation_features,
                irregular_features=irregular_features,
                future_queries=future_queries,
            )
        if self.fusion_type == "add_i2f_var":
            return self._forward_add_i2f_var(
                foundation_features=foundation_features,
                irregular_features=irregular_features,
                future_queries=future_queries,
            )
        return self._forward_film_i2f(
            foundation_features=foundation_features,
            irregular_features=irregular_features,
            future_queries=future_queries,
        )


class PredictionHead(nn.Module):
    """最终输出头：把融合后的隐藏表示映射为预测值。"""

    def __init__(self, hidden_dim: int, c_out: int, dropout: float, num_variables: int | None = None):
        super().__init__()
        self.c_out = c_out
        self.num_variables = num_variables
        self.output_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, c_out)
        self.variable_proj = nn.Linear(hidden_dim, 1)
        self.variable_output_proj = None
        if num_variables is not None and num_variables != c_out:
            self.variable_output_proj = nn.Linear(num_variables, c_out)

    def forward(self, fused_features: Tensor) -> Tensor:
        fused_features = self.output_dropout(fused_features)
        if fused_features.ndim == 4:
            predictions = self.variable_proj(fused_features).squeeze(-1)
            if self.variable_output_proj is not None:
                predictions = self.variable_output_proj(predictions)
            return predictions
        return self.proj(fused_features)
