from __future__ import annotations

"""Scale-aware routing blocks for the foundation branch."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .config import ScaleRoutingContext


def _zero_init_last_linear(module: nn.Sequential) -> None:
    last_linear = module[-1]
    if not isinstance(last_linear, nn.Linear):
        raise TypeError("Expected the last module to be nn.Linear.")
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    mask_f = mask.float()
    denom = mask_f.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return ((values * mask_f).sum(dim=dim, keepdim=True) / denom).squeeze(dim)


def _masked_std(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    mean = _masked_mean(values, mask, dim=dim)
    centered = values - mean.unsqueeze(dim)
    var = _masked_mean(centered.square(), mask, dim=dim)
    return torch.sqrt(var.clamp_min(1e-8))


def _masked_first(values: Tensor, mask: Tensor) -> Tensor:
    if mask.dim() == values.dim():
        mask = mask.squeeze(-1)
    has_valid = mask.any(dim=2)
    length = values.shape[2]
    index_grid = torch.arange(length, device=values.device).view(1, 1, length)
    filled = torch.where(mask, index_grid, torch.full_like(index_grid, length))
    first_idx = filled.argmin(dim=2).clamp(max=length - 1)
    gathered = values.gather(
        2,
        first_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, values.shape[-1]),
    ).squeeze(2)
    return torch.where(has_valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))


def _masked_last(values: Tensor, mask: Tensor) -> Tensor:
    if mask.dim() == values.dim():
        mask = mask.squeeze(-1)
    has_valid = mask.any(dim=2)
    length = values.shape[2]
    index_grid = torch.arange(length, device=values.device).view(1, 1, length)
    filled = torch.where(mask, index_grid, torch.full_like(index_grid, -1))
    last_idx = filled.argmax(dim=2).clamp(max=length - 1)
    gathered = values.gather(
        2,
        last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, values.shape[-1]),
    ).squeeze(2)
    return torch.where(has_valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    if mask.dim() == values.dim():
        mask = mask.squeeze(-1)
    has_valid = mask.any(dim=2, keepdim=True)
    expanded_mask = mask.unsqueeze(-1)
    neg_inf = torch.tensor(float("-inf"), device=values.device, dtype=values.dtype)
    masked_values = torch.where(expanded_mask, values, neg_inf)
    pooled = masked_values.amax(dim=2)
    return torch.where(has_valid, pooled, torch.zeros_like(pooled))


class ScaleStatsEncoder(nn.Module):
    """Summarize each variable sequence into segment-level scale descriptors."""

    def __init__(
        self,
        hidden_dim: int,
        num_segments: int,
        dropout: float,
        use_explicit_statistics: bool = True,
    ):
        super().__init__()
        if num_segments <= 0:
            raise ValueError("num_segments must be positive.")
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.use_explicit_statistics = use_explicit_statistics
        self.segment_value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.segment_stats_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.segment_norm = nn.LayerNorm(hidden_dim)
        _zero_init_last_linear(self.segment_stats_proj)

    def _pad_to_segments(
        self,
        sequence_features: Tensor,
        token_mask: Tensor,
        time_values: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, int]:
        batch_size, seq_len, hidden_dim = sequence_features.shape
        segment_len = max(1, math.ceil(seq_len / self.num_segments))
        target_len = segment_len * self.num_segments
        pad_len = target_len - seq_len
        if pad_len <= 0:
            return sequence_features, token_mask, time_values, segment_len

        feature_pad = torch.zeros(batch_size, pad_len, hidden_dim, device=sequence_features.device, dtype=sequence_features.dtype)
        mask_pad = torch.zeros(batch_size, pad_len, device=token_mask.device, dtype=token_mask.dtype)
        sequence_features = torch.cat([sequence_features, feature_pad], dim=1)
        token_mask = torch.cat([token_mask, mask_pad], dim=1)
        if time_values is not None:
            time_pad = torch.zeros(batch_size, pad_len, 1, device=time_values.device, dtype=time_values.dtype)
            time_values = torch.cat([time_values, time_pad], dim=1)
        return sequence_features, token_mask, time_values, segment_len

    def forward(
        self,
        sequence_features: Tensor,
        token_mask: Tensor,
        time_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(segment_descriptor, segment_mask, segment_stats)``.

        ``sequence_features`` is expected to exclude the variable-prompt token.
        """

        if token_mask.dim() == 3:
            token_mask = token_mask.squeeze(-1)
        if time_values is not None and time_values.dim() == 3:
            time_values = time_values.squeeze(-1).unsqueeze(-1)

        sequence_features, token_mask, time_values, segment_len = self._pad_to_segments(
            sequence_features=sequence_features,
            token_mask=token_mask,
            time_values=time_values,
        )
        batch_size, seq_len, hidden_dim = sequence_features.shape
        num_segments = seq_len // segment_len

        segment_features = sequence_features.contiguous().view(batch_size, num_segments, segment_len, hidden_dim)
        segment_mask = token_mask.contiguous().view(batch_size, num_segments, segment_len).bool()

        pooled_mask = segment_mask.unsqueeze(-1).float()
        pooled_denom = pooled_mask.sum(dim=2).clamp_min(1e-8)
        pooled = (segment_features * pooled_mask).sum(dim=2) / pooled_denom

        valid_ratio = segment_mask.float().mean(dim=2, keepdim=True)

        if time_values is None:
            mean_time = torch.zeros(batch_size, num_segments, 1, device=sequence_features.device, dtype=sequence_features.dtype)
            std_time = torch.zeros_like(mean_time)
            span_time = torch.zeros_like(mean_time)
        else:
            time_segments = time_values.contiguous().view(batch_size, num_segments, segment_len, 1)
            mean_time = _masked_mean(time_segments, segment_mask.unsqueeze(-1), dim=2)
            std_time = _masked_std(time_segments, segment_mask.unsqueeze(-1), dim=2)
            positive_inf = torch.tensor(float("inf"), device=time_segments.device, dtype=time_segments.dtype)
            negative_inf = torch.tensor(float("-inf"), device=time_segments.device, dtype=time_segments.dtype)
            time_min = torch.where(segment_mask.unsqueeze(-1), time_segments, positive_inf).amin(dim=2)
            time_max = torch.where(segment_mask.unsqueeze(-1), time_segments, negative_inf).amax(dim=2)
            span_time = (time_max - time_min).nan_to_num(0.0)
            has_valid = segment_mask.any(dim=2, keepdim=True)
            mean_time = mean_time.where(has_valid, torch.zeros_like(mean_time))
            std_time = std_time.where(has_valid, torch.zeros_like(std_time))
            span_time = span_time.where(has_valid, torch.zeros_like(span_time))

        segment_stats = torch.cat([valid_ratio, mean_time, std_time, span_time], dim=-1)
        if not self.use_explicit_statistics:
            segment_stats = torch.zeros_like(segment_stats)
        descriptor = self.segment_value_proj(pooled) + self.segment_stats_proj(segment_stats)
        descriptor = self.segment_norm(descriptor)
        segment_mask = segment_mask.any(dim=2)
        return descriptor, segment_mask, segment_stats


class ScaleRouter(nn.Module):
    """Produce expert weights from segment descriptors and optional context."""

    def __init__(self, hidden_dim: int, num_experts: int, router_hidden_dim: int, temperature: float, dropout: float):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.temperature = max(float(temperature), 1e-4)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 4, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, num_experts),
        )
        _zero_init_last_linear(self.router)

    def forward(
        self,
        segment_descriptor: Tensor,
        segment_mask: Tensor | None = None,
        future_context: Tensor | None = None,
        irregular_context: Tensor | None = None,
        variable_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if future_context is None:
            future_context = torch.zeros_like(segment_descriptor)
        if irregular_context is None:
            irregular_context = torch.zeros_like(segment_descriptor)
        if variable_context is None:
            variable_context = torch.zeros_like(segment_descriptor)

        router_input = torch.cat(
            [segment_descriptor, future_context, irregular_context, variable_context],
            dim=-1,
        )
        logits = self.router(router_input)
        if segment_mask is not None:
            if segment_mask.dim() == 3:
                segment_mask = segment_mask.squeeze(-1)
            logits = logits.masked_fill(~segment_mask.unsqueeze(-1), -1e8)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        if segment_mask is not None:
            weights = weights * segment_mask.unsqueeze(-1).float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return weights, logits


class ScaleExpertAdapter(nn.Module):
    """A lightweight residual expert used by the routing bank."""

    def __init__(self, hidden_dim: int, expert_hidden_dim: int, dropout: float):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, hidden_dim),
        )
        _zero_init_last_linear(self.adapter)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.adapter(x)


class ScaleExpertBank(nn.Module):
    """Apply several experts and mix them by router weights."""

    def __init__(self, hidden_dim: int, num_experts: int, expert_hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [ScaleExpertAdapter(hidden_dim=hidden_dim, expert_hidden_dim=expert_hidden_dim, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, segment_descriptor: Tensor, router_weights: Tensor) -> tuple[Tensor, Tensor]:
        expert_outputs = [expert(segment_descriptor) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=-2)
        if router_weights.shape[:-1] != stacked.shape[:-2]:
            raise ValueError(
                f"router_weights shape {tuple(router_weights.shape)} is incompatible with expert outputs {tuple(stacked.shape)}."
            )
        mixed = (stacked * router_weights.unsqueeze(-1)).sum(dim=-2)
        return mixed, stacked


class ScaleSegmentExpertBank(nn.Module):
    """Segment-level experts with different pooling biases for C2-lite."""

    def __init__(self, hidden_dim: int, num_experts: int, expert_hidden_dim: int, dropout: float):
        super().__init__()
        if num_experts != 3:
            raise ValueError("ScaleSegmentExpertBank expects exactly 3 experts: short, mid, long.")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        short_in_dim = hidden_dim * 3 + 4
        mid_in_dim = hidden_dim * 3 + 4
        long_in_dim = hidden_dim * 4 + 4
        self.short_expert = nn.Sequential(
            nn.Linear(short_in_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, hidden_dim),
        )
        self.mid_expert = nn.Sequential(
            nn.Linear(mid_in_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim * 1, hidden_dim),
        )
        self.long_expert = nn.Sequential(
            nn.Linear(long_in_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, hidden_dim),
        )
        for module in (self.short_expert, self.mid_expert, self.long_expert):
            _zero_init_last_linear(module)

    def forward(
        self,
        segment_tokens: Tensor,
        segment_mask: Tensor,
        segment_descriptor: Tensor,
        segment_stats: Tensor,
        router_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if segment_mask.dim() != 3:
            raise ValueError("ScaleSegmentExpertBank expects token-level segment_mask with shape [B, S, L].")
        token_mask = segment_mask.unsqueeze(-1)
        mean_token = _masked_mean(segment_tokens, token_mask, dim=2)
        std_token = _masked_std(segment_tokens, token_mask, dim=2)
        first_token = _masked_first(segment_tokens, segment_mask)
        last_token = _masked_last(segment_tokens, segment_mask)
        max_token = _masked_max(segment_tokens, segment_mask)

        short_input = torch.cat([segment_descriptor, last_token, mean_token, segment_stats], dim=-1)
        mid_input = torch.cat([segment_descriptor, mean_token, std_token, segment_stats], dim=-1)
        long_input = torch.cat([segment_descriptor, mean_token, max_token, first_token, segment_stats], dim=-1)

        short_output = self.short_expert(short_input)
        mid_output = self.mid_expert(mid_input)
        long_output = self.long_expert(long_input)

        stacked = torch.stack([short_output, mid_output, long_output], dim=-2)
        if router_weights.shape[:-1] != stacked.shape[:-2]:
            raise ValueError(
                f"router_weights shape {tuple(router_weights.shape)} is incompatible with expert outputs {tuple(stacked.shape)}."
            )
        mixed = (stacked * router_weights.unsqueeze(-1)).sum(dim=-2)
        mixed = mixed * segment_mask.any(dim=2, keepdim=True).float()
        return mixed, stacked


def routing_entropy(router_weights: Tensor) -> Tensor:
    """Compute per-sample routing entropy for diagnostics."""
    probs = router_weights.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1)


def summarize_router_usage(router_weights: Tensor) -> Tensor:
    """Average expert usage across batch and segments."""
    return router_weights.mean(dim=tuple(range(router_weights.dim() - 1)))


__all__ = [
    "ScaleRoutingContext",
    "ScaleStatsEncoder",
    "ScaleRouter",
    "ScaleExpertAdapter",
    "ScaleExpertBank",
    "ScaleSegmentExpertBank",
    "routing_entropy",
    "summarize_router_usage",
]
